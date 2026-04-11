# -*- coding: utf-8 -*-
from __future__ import annotations

from fastapi.responses import JSONResponse


class UTF8JSONResponse(JSONResponse):
    # Force explicit UTF-8 charset in Content-Type
    media_type = "application/json; charset=utf-8"

