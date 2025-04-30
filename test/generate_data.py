import random
NUM_HOSTS = 20

def generate_test_flows(total_flows=1000, repeat_percentage=0.8, num_hosts=NUM_HOSTS):
    # Create list of hosts
    hosts = [f"{i}" for i in range(1, num_hosts + 1)]
    
    # Generate all possible host-destination pairs
    all_pairs = [(h1, h2) for h1 in hosts for h2 in hosts if h1 != h2]
    
    # Select 20% of pairs to repeat
    num_repeat_pairs = int(len(all_pairs) * 0.2)
    repeat_pairs = random.sample(all_pairs, num_repeat_pairs)
    
    # Generate flows
    flows = []
    
    # 80% flows with repeating pairs
    repeat_flows = int(total_flows * repeat_percentage)
    for _ in range(repeat_flows):
        src, dst = random.choice(repeat_pairs)
        bandwidth = random.randint(2, 5)
        duration = random.randint(1, 2)
        flows.append(f"{src} {dst} {bandwidth} {duration}")
    
    # 20% flows with non-repeating pairs
    remaining_pairs = [p for p in all_pairs if p not in repeat_pairs]
    for _ in range(total_flows - repeat_flows):
        if remaining_pairs:
            src, dst = random.choice(remaining_pairs)
            bandwidth = random.randint(1, 3)
            flows.append(f"{src} {dst} {bandwidth} 1")
            remaining_pairs.remove((src, dst))
    
    # Shuffle the flows
    random.shuffle(flows)
    
    return flows

# Generate and save flows to file
flows = generate_test_flows(num_hosts=NUM_HOSTS)
with open(f'{NUM_HOSTS}_hosts_test.txt', 'w') as f:
    for flow in flows:
        f.write(flow + '\n')