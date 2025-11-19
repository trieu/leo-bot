import asyncio
import json
import redis.asyncio as redis

import structlog
from leocdp.cdp_client import CDPClient
from leocdp.profile_queue import LEOCDP_PROFILE_QUEUE_NAME
from main_config import REDIS_HOST, REDIS_PORT

logger = structlog.get_logger(__name__)

REDIS_URL = f"redis://{REDIS_HOST}:{int(REDIS_PORT)}"


class ProfileJobWorker:
    def __init__(self, redis_url=REDIS_URL):
        self.redis = redis.from_url(redis_url)
        self.client = CDPClient()

    async def run(self):
        logger.info("worker.start", message="Profile Worker started")

        while True:
            try:
                
                msg = await self.redis.brpop(LEOCDP_PROFILE_QUEUE_NAME)
            except Exception as e:
                logger.error("worker.redis_error", error=str(e))
                await asyncio.sleep(1)
                continue

            _, payload_raw = msg
            profile_dict = json.loads(payload_raw)

            logger.info("worker.job_received", job=profile_dict)

            try:
                result = await self.client.save_profile(profile_dict)
                logger.info("worker.job_success", result=result)

            except Exception as ex:
                logger.error("worker.job_failed", error=str(ex), job=profile_dict)

            await asyncio.sleep(0.05)
