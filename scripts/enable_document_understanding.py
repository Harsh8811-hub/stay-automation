"""
Script to enable document_understanding in MongoDB for hotel 111111 and invalidate cache.
"""
import asyncio
import os
import sys
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("enable_document_understanding")

async def enable_for_hotel(target_hotel_id: int = 111111):
    from app.db.mongodb import connect_mongo, close_mongo, hotel_settings_collection, get_database
    from app.services.hotel_config_service import get_hotel_config, invalidate_cache
    
    await connect_mongo()
    try:
        coll = hotel_settings_collection()
        
        # 1. Inspect existing document
        query = {"hotel_id": str(target_hotel_id)}
        doc = await coll.find_one(query)
        
        logger.info(f"Target Hotel: {target_hotel_id}")
        if doc:
            logger.info(f"Found document in '{coll.name}' with ID: {doc.get('_id')}")
            current_behavior = doc.get("behavior") or {}
            current_features = current_behavior.get("features") or {}
            logger.info(f"Current features in MongoDB: {current_features}")
        else:
            logger.warning(f"No document found for hotel {target_hotel_id} in '{coll.name}'. Checking fallback collections...")
            # Also check Automation-AI-Setting vs hotel_settings
            alt_coll = get_database()["hotel_settings"]
            doc = await alt_coll.find_one(query)
            if doc:
                coll = alt_coll
                logger.info(f"Found document in 'hotel_settings' with ID: {doc.get('_id')}")
        
        if doc:
            # 2. Update document with documentUnderstandingEnabled and documentUnderstanding = True
            update_res = await coll.update_many(
                query,
                {
                    "$set": {
                        "behavior.features.documentUnderstandingEnabled": True,
                        "behavior.features.documentUnderstanding": True,
                        "behavior.features.document_understanding": True,
                    }
                }
            )
            logger.info(f"MongoDB Update Result: matched={update_res.matched_count}, modified={update_res.modified_count}")
        else:
            logger.info(f"Creating new setting document for hotel {target_hotel_id}...")
            new_doc = {
                "hotel_id": str(target_hotel_id),
                "aiEnabled": True,
                "behavior": {
                    "conversationTone": "professional",
                    "responseLength": 40,
                    "features": {
                        "aiEnabled": True,
                        "directBooking": True,
                        "documentUnderstandingEnabled": True,
                        "documentUnderstanding": True,
                        "document_understanding": True,
                    }
                }
            }
            await coll.insert_one(new_doc)
            logger.info(f"Created new document for hotel {target_hotel_id}")

        # 3. Invalidate cache
        invalidate_cache(target_hotel_id)
        logger.info(f"Config cache invalidated for hotel {target_hotel_id}")

        # 4. Verify by loading config
        loaded_config = await get_hotel_config(target_hotel_id)
        is_enabled = loaded_config.business_config.features.document_understanding_enabled
        logger.info(f"Verification Check: document_understanding_enabled = {is_enabled}")
        
        from app.services.feature_gate import FeatureGate
        gate = FeatureGate(loaded_config)
        gate_res = gate.check_document_understanding()
        logger.info(f"FeatureGate Check: allowed={gate_res.allowed}, response={gate_res.response}")
        
        return is_enabled and gate_res.allowed

    finally:
        await close_mongo()

if __name__ == "__main__":
    success = asyncio.run(enable_for_hotel(111111))
    print(f"\nResult: {'SUCCESS' if success else 'FAILED'}")
