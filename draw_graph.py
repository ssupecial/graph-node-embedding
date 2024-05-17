import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from node2vec import Node2Vec
import time
from grakel import Graph
from grakel.kernels import WeisfeilerLehman


def generate_instance(num_jobs, num_machines, time_min=1, time_max=100):

    np.random.seed(0)  # Seed for reproducibility

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
    G = make_graph(processing_time_matrix, machine_matrix)

    # print("Nodes:", G.nodes(data=True))
    # print("Edges:", G.edges(data=True))
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


def edge_similarity(graph1, graph2):
    cur = time.time()
    edges1 = set(graph1.edges())
    edges2 = set(graph2.edges())

    common_edges = edges1.intersection(edges2)
    all_edges = edges1.union(edges2)

    if len(edges1) == 0:
        return 0.0

    similarity = len(common_edges) / len(all_edges)
    fin = time.time()
    return similarity, fin - cur


def edge_similarity_grakel(
    source_graph: nx.DiGraph, target_graph: nx.DiGraph, n_iter: int
):
    cur = time.time()

    def nx_to_grakel(G):
        node_labels = nx.get_node_attributes(G, "machine")  # node attribute
        edge_labels = nx.get_edge_attributes(G, "type")  # edge attribute
        return Graph(list(G.edges()), node_labels=node_labels, edge_labels=edge_labels)

    wl_kernel = WeisfeilerLehman(n_iter=n_iter)
    K = wl_kernel.fit_transform(
        [nx_to_grakel(source_graph), nx_to_grakel(target_graph)]
    )

    fin = time.time()

    similarity = K[0, 1]  # Source-Target graph similarity
    ratio = K[0, 0]  # Source graph similarity
    return similarity / ratio, fin - cur


def edge_similarity_cosine(source_graph: nx.DiGraph, target_graph: nx.DiGraph):
    pass


def edge_similarity_mcs(source_graph: nx.DiGraph, target_graph: nx.DiGraph):
    # Maximum Common Subgraph
    cur = time.time()
    mcs = nx.algorithms.isomorphism.GraphMatcher(source_graph, target_graph)
    max_common_subgraph = max(
        (len(subgraph), subgraph) for subgraph in mcs.subgraph_isomorphisms_iter()
    )[1]
    fin = time.time()

    print("Maximum Common Subgraph:", max_common_subgraph)
    return len(max_common_subgraph) / len(source_graph.edges()), fin - cur


def edge_similarity_sc(source_graph: nx.DiGraph, target_graph: nx.DiGraph):
    # Spectral Comparison
    cur = time.time()
    A1 = nx.adjacency_matrix(source_graph).todense()
    A2 = nx.adjacency_matrix(target_graph).todense()

    # 고유값 계산
    eigenvalues1 = np.linalg.eigvals(A1)
    eigenvalues2 = np.linalg.eigvals(A2)

    # 유클리드 거리 계산
    euclidean_distance = np.linalg.norm(eigenvalues1 - eigenvalues2)
    # print("Euclidean Distance:", euclidean_distance)

    # 코사인 유사도 계산
    cosine_similarity = np.dot(eigenvalues1, eigenvalues2) / (
        np.linalg.norm(eigenvalues1) * np.linalg.norm(eigenvalues2)
    )
    fin = time.time()

    # return cosine_similarity, fin - cur
    return cosine_similarity


def make_vectors(processing_time_matrix, machine_matrix, dimensions=20):
    G = make_graph(processing_time_matrix, machine_matrix)
    node2vec = Node2Vec(
        G, dimensions=dimensions, walk_length=30, num_walks=200, workers=4, p=1, q=1
    )
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    node_embeddings = np.array([model.wv[str(node)] for node in G.nodes()])
    return node_embeddings


if __name__ == "__main__":

    # Parameters
    num_jobs = 100  # number of jobs
    num_machines = 20  # number of machines

    # Generate instance
    processing_time_matrix, machine_matrix = generate_instance(num_jobs, num_machines)
    embeddings = make_vectors(processing_time_matrix, machine_matrix, 20)

    exit()
    G = make_graph(processing_time_matrix, machine_matrix)

    # deprecated_time_matrix, deprecated_machine_matrix = random_mask(
    #     processing_time_matrix, machine_matrix, num_jobs, num_machines, 8, 8
    # )
    # deprecated_graph = make_graph(deprecated_time_matrix, deprecated_machine_matrix)

    # processing_time_matrix2, machine_matrix2 = generate_instance(num_jobs, num_machines)
    # G2 = make_graph(processing_time_matrix2, machine_matrix2)
    # draw_graph(G, num_machines)

    # print_instance(processing_time_matrix, machine_matrix)
    # print_instance(deprecated_time_matrix, deprecated_machine_matrix)

    # print(nx.graph_edit_distance(G, g))
    # print(nx.optimize_edit_paths(G, g))

    # Random mask를 통해서 instance 데이터 축소 10x10 -> 8x8
    # deprecated_time_matrix, dprecated_machine_matrix = random_mask(
    #     processing_time_matrix, machine_matrix, num_jobs, num_machines, 8, 8
    # )

    # Print instance
    # print_instance(processing_time_matrix, machine_matrix)
    # print_instance(deprecated_time_matrix, dprecated_machine_matrix)

    # Make Graph from instance
    # G = make_graph(processing_time_matrix, machine_matrix)
    # G = make_graph(deprecated_time_matrix, dprecated_machine_matrix)

    # print("Nodes:", G.nodes(data=True))
    # print("Edges:", G.edges(data=True))
    # draw_graph(G, num_machines)

    # similarity, duration = edge_similarity_sc(G, G2)
    # print(similarity)
    # print(duration)

    # result, duration = edge_similarity_grakel(G, deprecated_graph, 10)
    # print(result)
    # print(duration)

    # exit()

    cur2 = time.time()

    node2vec = Node2Vec(
        G, dimensions=20, walk_length=30, num_walks=200, workers=4, p=1, q=1
    )

    ## if d_graph is big enough to fit in the memory, pass temp_folder which has enough disk space
    # Note: It will trigger "sharedmem" in Parallel, which will be slow on smaller graphs
    # node2vec = Node2Vec(graph, dimensions=64, walk_length=30, num_walks=200, workers=4, temp_folder="/mnt/tmp_data")

    # Embed
    model = node2vec.fit(
        window=10, min_count=1, batch_words=4
    )  # Any keywords acceptable by gensim.Word2Vec can be passed, `dimensions` and `workers` are automatically passed (from the Node2Vec constructor)

    # Look for most similar nodes
    fin2 = time.time()
    a = model.wv.most_similar("1-1")  # Output node names are always strings
    print(a)
    print(fin2 - cur2)
    EMBEDDING_FILENAME = "test.emb"
    EMBEDDING_MODEL_FILENAME = "test.model"

    # Save embeddings for later use
    model.wv.save_word2vec_format(EMBEDDING_FILENAME)

    # Save model for later use
    model.save(EMBEDDING_MODEL_FILENAME)

    node_embeddings = np.array([model.wv[str(node)] for node in G.nodes()])
    print(node_embeddings)
    print("Shape")
    print(node_embeddings.shape)
