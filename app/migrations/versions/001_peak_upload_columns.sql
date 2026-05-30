-- Track peak image upload status and prune successfully uploaded edge copies.
ALTER TABLE minute_peak_snapshot ADD COLUMN IF NOT EXISTS uploaded_to_server VARCHAR DEFAULT 'Pending';
ALTER TABLE minute_peak_snapshot ADD COLUMN IF NOT EXISTS server_path VARCHAR;
ALTER TABLE minute_peak_snapshot ADD COLUMN IF NOT EXISTS uploaded_at TIMESTAMPTZ;
