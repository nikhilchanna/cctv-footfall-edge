-- Peak upload retry tracking (backoff + max attempts).
ALTER TABLE minute_peak_snapshot ADD COLUMN IF NOT EXISTS upload_retry_ctr INTEGER DEFAULT 0;
ALTER TABLE minute_peak_snapshot ADD COLUMN IF NOT EXISTS last_upload_attempt TIMESTAMPTZ;
