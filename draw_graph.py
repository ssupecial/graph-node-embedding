import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":

    N = 2
    M = 15
    # 데이터셋 로드 (여기서는 직접 값을 넣습니다)
    processing_time_matrix = np.array(
        [
            [25, 75, 75, 76, 38, 62, 38, 59, 14, 13, 46, 31, 57, 92, 3],
            [67, 5, 11, 11, 40, 34, 77, 42, 35, 96, 22, 55, 21, 29, 16],
            # 여기에 더 많은 데이터를 추가할 수 있습니다.
        ]
    )
    machine_matrix = np.array(
        [
            [4, 12, 15, 2, 11, 3, 5, 8, 1, 13, 6, 10, 7, 14, 9],
            [6, 1, 4, 9, 5, 2, 13, 15, 7, 8, 11, 3, 10, 14, 12],
            # 여기에 더 많은 데이터를 추가할 수 있습니다.
        ]
    )

    # 그래프 초기화
    G = nx.DiGraph()

    # Conjunctive Graph
    for job_index, (time_row, machine_row) in enumerate(
        zip(processing_time_matrix, machine_matrix)
    ):
        previous_node = None
        for step_index, (time, machine) in enumerate(zip(time_row, machine_row)):
            # 노드 추가
            node = (job_index, step_index)
            G.add_node(node, machine=machine, time=time)

            # 순차적 간선 추가 (동일 작업 내)
            if previous_node:
                G.add_edge(previous_node, node, type="sequential")
            previous_node = node

            # 작업 간 연결 추가 (동일 기계 사용)

    # Disjunctive Graph
    machine_indexes = set(machine_matrix.flatten().tolist())
    for m_idx in machine_indexes:
        job_ids, step_ids = np.where(machine_matrix == m_idx)

        for job_id, step_id in zip(job_ids, step_ids):
            node = (job_id, step_id)
            for job_id2, step_id2 in zip(job_ids, step_ids):
                if job_id != job_id2 and step_id != step_id2:
                    other_node = (job_id2, step_id2)
                    G.add_edge(other_node, node, type="inter-job")

    print("Nodes:", G.nodes(data=True))
    print("Edges:", G.edges(data=True))
    color_map = plt.get_cmap(
        "tab20"
    )  # 20가지 색상 중 선택 (tab20은 20개 색상을 가진 색상 맵)
    colors = [color_map(i / 15) for i in range(15)]

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
