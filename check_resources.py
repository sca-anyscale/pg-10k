import ray
ray.init()
nodes = ray.nodes()
alive = [n for n in nodes if n["Alive"]]
head_ip = ray.get_runtime_context().worker.node_ip_address
workers = [n for n in alive if n["NodeManagerAddress"] != head_ip]
head = [n for n in alive if n["NodeManagerAddress"] == head_ip]
print(f"Total alive: {len(alive)}, workers: {len(workers)}")
print(f"\nHead node:")
for h in head:
    res = h["Resources"]
    print(f"  CPU={res.get('CPU',0)} fake_GPU={res.get('fake_GPU',0)}")
print(f"\nFirst 3 workers:")
for w in workers[:3]:
    res = w["Resources"]
    nid = w["NodeID"][:8]
    print(f"  Node {nid}: CPU={res.get('CPU',0)} fake_GPU={res.get('fake_GPU',0)}")
