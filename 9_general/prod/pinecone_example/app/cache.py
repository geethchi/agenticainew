from datetime import datetime, timedelta

# In-memory cache
cache = {}
CACHE_EXPIRATION = timedelta(minutes=5)  # Cache expiry time

def get_cached_response(query):
    """
    Retrieves a cached response if available and not expired.
    """
    cache_entry = cache.get(query)
    if cache_entry:
        response, timestamp = cache_entry
        if datetime.now() - timestamp < CACHE_EXPIRATION:
            return response
    return None

def set_cached_response(query, response):
    """
    Stores the response in cache with the current timestamp.
    """
    cache[query] = (response, datetime.now())
