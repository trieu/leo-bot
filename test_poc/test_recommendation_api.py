from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any

import asyncio
from async_pgvector_recommend import (
    create_pool,
    ensure_extension_and_tables,
    create_hnsw_index_for_products,
    upsert_profile,
    upsert_product,
    batch_upsert_profiles,
    batch_upsert_products,
    recommend_products_for_profile,
)

# -----------------------------------
# FastAPI initialization
# -----------------------------------
app = FastAPI(title="CDP Recommendation API (PGVector)", version="2.0")

# Connection pool reference
pool = None


# -----------------------------------
# Pydantic models for input validation
# -----------------------------------
class ProfileRequest(BaseModel):
    profile_id: str
    page_view_keywords: List[str]
    purchase_keywords: List[str]
    interest_keywords: List[str]
    additional_info: Dict[str, Any] = {}
    max_recommendation_size: int = Field(8, description="Default top N recommendations")
    except_product_ids: List[str] = []


class ProductRequest(BaseModel):
    product_id: str
    product_name: str
    product_category: str
    product_keywords: List[str]
    additional_info: Dict[str, Any] = {}


# -----------------------------------
# Lifecycle events
# -----------------------------------
@app.on_event("startup")
async def on_startup():
    global pool
    pool = await create_pool()
    await ensure_extension_and_tables(pool)
    await create_hnsw_index_for_products(pool)
    print("✅ Database initialized and connection pool ready.")


@app.on_event("shutdown")
async def on_shutdown():
    global pool
    if pool:
        await pool.close()
        print("🛑 Connection pool closed.")


# -----------------------------------
# Default route
# -----------------------------------
@app.get("/")
async def index():
    return {"message": "CDP Recommendation API using PostgreSQL + pgvector"}


# -----------------------------------
# Profile Endpoints
# -----------------------------------
@app.post("/add-profile/")
async def api_add_profile(profile: ProfileRequest):
    try:
        await upsert_profile(
            pool,
            profile.profile_id,
            profile.page_view_keywords,
            profile.purchase_keywords,
            profile.interest_keywords,
            profile.additional_info,
        )
        return {"status": "Profile added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Add profile failed: {e}")


@app.post("/add-profiles/")
async def api_add_profiles(profiles: List[ProfileRequest]):
    try:
        inserted = await batch_upsert_profiles(
            pool,
            [
                {
                    "profile_id": p.profile_id,
                    "page_view_keywords": p.page_view_keywords,
                    "purchase_keywords": p.purchase_keywords,
                    "interest_keywords": p.interest_keywords,
                    "additional_info": p.additional_info,
                }
                for p in profiles
            ],
        )
        return {"status": f"{inserted} profiles added/updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch profile upsert failed: {e}")


# -----------------------------------
# Product Endpoints
# -----------------------------------
@app.post("/add-product/")
async def api_add_product(product: ProductRequest):
    try:
        await upsert_product(
            pool,
            product.product_id,
            product.product_name,
            product.product_category,
            product.product_keywords,
            product.additional_info,
        )
        return {"status": "Product added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Add product failed: {e}")


@app.post("/add-products/")
async def api_add_products(products: List[ProductRequest]):
    try:
        inserted = await batch_upsert_products(
            pool,
            [
                {
                    "product_id": p.product_id,
                    "name": p.product_name,
                    "category": p.product_category,
                    "keywords": p.product_keywords,
                    "additional_info": p.additional_info,
                }
                for p in products
            ],
        )
        return {"status": f"{inserted} products added/updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch product upsert failed: {e}")


# -----------------------------------
# Recommendation Endpoint
# -----------------------------------
@app.post("/recommend/")
async def api_recommend(profile: ProfileRequest):
    """
    Add/update profile, then get recommendations in real-time.
    """
    try:
        await upsert_profile(
            pool,
            profile.profile_id,
            profile.page_view_keywords,
            profile.purchase_keywords,
            profile.interest_keywords,
            profile.additional_info,
        )
        result = await recommend_products_for_profile(
            pool,
            profile.profile_id,
            profile.max_recommendation_size,
            profile.except_product_ids,
        )
        if not result or not result.get("recommended_products"):
            raise HTTPException(status_code=404, detail="No recommendations found")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {e}")


@app.get("/recommend/{profile_id}")
async def api_get_recommend(profile_id: str, top_n: int = 8, except_product_ids: str = ""):
    try:
        ids = [x for x in except_product_ids.split(",") if x]
        result = await recommend_products_for_profile(pool, profile_id, top_n, ids)
        if not result or not result.get("recommended_products"):
            raise HTTPException(status_code=404, detail="No recommendations found")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {e}")


# -----------------------------------
# Run app manually (dev mode)
# -----------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_pgvector_recommend:app", host="0.0.0.0", port=8000, reload=True)
