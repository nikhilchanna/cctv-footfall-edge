"""Shared error logging helper (avoids circular imports between tasks and peak_upload)."""

import logging

logger = logging.getLogger(__name__)


def report_internal_error(source: str, error_msg: str, traceback_str: str = None):
    logger.error(f"Error from {source}: {error_msg} | Traceback: {traceback_str}")
