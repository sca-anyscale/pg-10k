#!/usr/bin/env python3

import os
from pprint import pprint
import ray

address = os.environ.get("RAY_ADDRESS", "auto")
addr = ray.init(address=address)
pprint(ray.available_resources())
