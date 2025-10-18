-- https://chatgpt.com/share/68f373d5-e060-8000-9816-2f94e37c7ecf

-- Find top 10 similar profiles to a given profile using cosine (pgvector):
SELECT customer_id, profile->>'name' AS name, profile_embedding <=> target.profile_embedding AS distance
FROM customer_profile target, customer_profile
WHERE target.customer_id = 'customer-123' AND customer_profile.customer_id != target.customer_id
ORDER BY profile_embedding <=> target.profile_embedding
LIMIT 10;

-- Get churn-risk customers (experience_score <= -40):
SELECT customer_id, tenant_id, experience_score, segment_reason
FROM customer_metrics
WHERE experience_score <= -40;
