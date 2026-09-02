#!/usr/bin/env python3

from pprint import pprint
import ray

ray.init()
pprint(ray.available_resources())
