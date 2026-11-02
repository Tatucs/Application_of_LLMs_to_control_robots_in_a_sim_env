import time

from aiolimiter import AsyncLimiter
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.models.wrapper import WrapperModel


class RateLimitedModel(WrapperModel):
    """Model wrapper that waits for the limiter before every LLM request.

    One agent.run makes several model requests (one per tool-call round),
    so the limit is applied here, not around agent.run.
    Give each provider quota its own AsyncLimiter; agents using the same
    model share one limiter instance.

    AsyncLimiter is a leaky bucket: it allows a burst of max_rate, then
    drains at max_rate/time_period. Worst case in a 60 s window is
    max_rate + 60 * max_rate / time_period, so keep max_rate small.
    """

    def __init__(self, wrapped: Model | KnownModelName | str, limiter: AsyncLimiter):
        super().__init__(wrapped)
        self.limiter = limiter

    async def request(self, *args, **kwargs):
        async with self.limiter:
            print(f"[rate] LLM request {time.strftime('%X')}")
            return await self.wrapped.request(*args, **kwargs)
