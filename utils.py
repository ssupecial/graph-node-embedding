import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def generate_instance(num_jobs, num_machines, time_min=1, time_max=100, random_seed=0):

    np.random.seed(random_seed)  # Seed for reproducibility

    # Initialize arrays
    times = np.zeros((num_jobs, num_machines), dtype=int)
    machines = np.zeros((num_jobs, num_machines), dtype=int)

    for i in range(num_jobs):
        # Generate non-duplicate times for each job
        times[i] = np.random.choice(
            range(time_min, time_max + 1), size=num_machines, replace=False
        )
        # Generate non-duplicate machines for each job
        machines[i] = np.random.permutation(range(1, num_machines + 1))

    return times, machines


def print_instance(times, machines):
    num_jobs, num_machines = times.shape

    print("Times")
    for j in range(num_jobs):
        for m in range(num_machines):
            print(f"{times[j, m]:3d}", end=" ")
        print()

    print("\nMachines")
    for j in range(num_jobs):
        for m in range(num_machines):
            print(f"{machines[j, m]:3d}", end=" ")
        print()


def make_graph(processing_time_matrix, machine_matrix) -> nx.DiGraph:
    G = nx.DiGraph()

    # Conjunctive Graph
    for job_index, (time_row, machine_row) in enumerate(
        zip(processing_time_matrix, machine_matrix)
    ):
        previous_node = None
        for step_index, (time, machine) in enumerate(zip(time_row, machine_row)):
            # 노드 추가
            node = f"{job_index}-{step_index}"
            G.add_node(node, machine=machine, time=time)

            # Conjunctive Graph (동일 작업 내)
            if previous_node:
                G.add_edge(previous_node, node, type="CONJUNCTIVE")
            previous_node = node

    # Disjunctive Graph (동일 기계 사용)
    machine_indexes = set(machine_matrix.flatten().tolist())
    for m_idx in machine_indexes:
        job_ids, step_ids = np.where(machine_matrix == m_idx)

        for job_id, step_id in zip(job_ids, step_ids):
            node = f"{job_id}-{step_id}"
            for job_id2, step_id2 in zip(job_ids, step_ids):
                if not (job_id == job_id2 and step_id == step_id2):
                    other_node = f"{job_id2}-{step_id2}"
                    G.add_edge(other_node, node, type="DISJUNCTIVE")

    return G


def draw_graph(G: nx.DiGraph, machine_num: int):
    color_map = plt.get_cmap(
        "tab20"
    )  # 20가지 색상 중 선택 (tab20은 20개 색상을 가진 색상 맵)
    colors = [color_map(i / machine_num) for i in range(machine_num)]
    # 노드 색상 배열 생성
    node_colors = [colors[G.nodes[node]["machine"] - 1] for node in G.nodes]
    fig = plt.figure(figsize=(8, 8))
    nx.draw(
        G,
        with_labels=True,
        node_color=node_colors,
        node_size=500,
        edge_color="#FF5733",
        linewidths=1,
        font_size=15,
    )
    plt.show()


def random_mask(
    processing_time_matrix,
    machine_matrix,
    num_job,  # job 개수
    num_machine,  # machine 개수
    decrement_num_job,  # 축소시킬 job 사이즈
    decrement_num_machine,  # 각 job의 축소시킬 operation 사이즈
):
    random_jobs = np.sort(
        np.random.choice(range(num_job), decrement_num_job, replace=False)
    )

    deprecated_time_matrix = processing_time_matrix[random_jobs]
    deprecated_machine_matrix = machine_matrix[random_jobs]

    start_machine = num_machine - decrement_num_machine
    deprecated_time_matrix = deprecated_time_matrix[:, start_machine:]
    deprecated_machine_matrix = deprecated_machine_matrix[:, start_machine:]

    return deprecated_time_matrix, deprecated_machine_matrix
