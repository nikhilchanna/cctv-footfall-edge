import logging
import requests
from datetime import datetime, timedelta
import pytz
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.models import DataTracker
from app.peak_upload import peak_upload_job
from app.error_reporting import report_internal_error

logger = logging.getLogger(__name__)

# Dummy external server URL. This should ideally come from CctvConfig or env var.
EXTERNAL_API_URL = "http://localhost:8081/api/v1/footfall-data"


def api_calling_thread_job():
    """Reads 10 Pending entries, marks In-progress, sends to API, updates status."""
    db: Session = SessionLocal()
    try:
        # Fetch 10 pending entries
        entries = db.query(DataTracker).filter(DataTracker.data_to_server_ack == "Pending").limit(10).all()
        if not entries:
            return

        # Mark as In-progress
        for entry in entries:
            entry.data_to_server_ack = "In-progress"
        db.commit()

        # Try API Call
        payload = [{
            "id": e.id,
            "cctvid": e.cctvid,
            "ctr_in": e.ctr_in,
            "ctr_out": e.ctr_out,
            "starttime": e.starttime.isoformat() if e.starttime else None,
            "endtime": e.endtime.isoformat() if e.endtime else None
        } for e in entries]

        try:
            # Making the external API call
            response = requests.post(EXTERNAL_API_URL, json=payload, timeout=10)
            is_success = response.status_code in (200, 201)
        except Exception as e:
            is_success = False
            logger.warning(f"API Call failed: {e}")

        # Update DB based on response
        for entry in entries:
            entry.data_to_server_ack = "Successful" if is_success else "Failed"
            entry.api_call_ctr += 1
            entry.last_api_call = datetime.utcnow()
        db.commit()

    except Exception as e:
        db.rollback()
        report_internal_error("ApiCallingThread", str(e))
    finally:
        db.close()

def retry_failed_thread_job():
    """Reads Failed entries, retries API call, updates status."""
    db: Session = SessionLocal()
    try:
        # We can also limit this so it doesn't process too many at once
        entries = db.query(DataTracker).filter(DataTracker.data_to_server_ack == "Failed").limit(10).all()
        if not entries:
            return

        payload = [{
            "id": e.id,
            "cctvid": e.cctvid,
            "ctr_in": e.ctr_in,
            "ctr_out": e.ctr_out,
            "starttime": e.starttime.isoformat() if e.starttime else None,
            "endtime": e.endtime.isoformat() if e.endtime else None
        } for e in entries]

        try:
            response = requests.post(EXTERNAL_API_URL, json=payload, timeout=10)
            is_success = response.status_code in (200, 201)
        except Exception as e:
            is_success = False
            logger.warning(f"Retry API Call failed: {e}")

        for entry in entries:
            entry.data_to_server_ack = "Successful" if is_success else "Failed"
            entry.api_call_ctr += 1
            entry.last_api_call = datetime.utcnow()
        db.commit()

    except Exception as e:
        db.rollback()
        report_internal_error("RetryFailedThread", str(e))
    finally:
        db.close()

def daily_cleanup_job():
    """Deletes successful entries from yesterday 24 hours (00:00 AM to 11:59 PM IST)."""
    db: Session = SessionLocal()
    try:
        ist_tz = pytz.timezone('Asia/Kolkata')
        now_ist = datetime.now(ist_tz)
        
        # Calculate yesterday's boundaries in IST
        yesterday_ist = now_ist - timedelta(days=1)
        start_of_yesterday_ist = yesterday_ist.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_yesterday_ist = yesterday_ist.replace(hour=23, minute=59, second=59, microsecond=999999)
        
        deleted_count = db.query(DataTracker).filter(
            DataTracker.data_to_server_ack == "Successful",
            DataTracker.createdAt >= start_of_yesterday_ist,
            DataTracker.createdAt <= end_of_yesterday_ist
        ).delete(synchronize_session=False)
        
        db.commit()
        logger.info(f"Cleanup Task: Deleted {deleted_count} successful entries from yesterday.")

    except Exception as e:
        db.rollback()
        report_internal_error("DailyCleanupThread", str(e))
    finally:
        db.close()
