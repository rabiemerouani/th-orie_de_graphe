import sys
import pandas as pd
import networkx as nx
import numpy as np
from io import StringIO
import matplotlib.pyplot as plt
from heapq import heappop, heappush
import heapq
import random
import copy
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QPushButton,
    QWidget,
    QLineEdit,
    QFileDialog,
    QMessageBox,
    QFrame,
    QHBoxLayout,
    QCheckBox,
    QLabel,
)
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QUrl
from pyvis.network import Network
from collections import deque
import copy
import csv  

class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.webview = QWebEngineView(self)
        self.webview.setMaximumSize(0, 0)
        self.initUI()

    def initUI(self):
        self.setWindowTitle("GRAPHEUR")
        self.setGeometry(100, 100, 800, 600)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        global main_layout
        main_layout = QHBoxLayout(central_widget)

        button_widget = QWidget()
        button_layout = QVBoxLayout(button_widget)

        self.directed = QCheckBox("Orientée")
        self.directed.toggled.connect(lambda: self.toggle_checkbox(self.directed))

        self.undirected = QCheckBox("Non Orientée")
        self.undirected.setChecked(True)
        self.undirected.toggled.connect(lambda: self.toggle_checkbox(self.undirected))

        self.checkbox_layout = QHBoxLayout()
        self.checkbox_layout.addWidget(self.directed)
        self.checkbox_layout.addWidget(self.undirected)
        button_layout.addLayout(self.checkbox_layout)

        self.load_csv_button = QPushButton("     Uploder Graph", self)
        self.load_csv_button.setIcon(QIcon("icons/graph.png"))
        self.load_csv_button.clicked.connect(self.loadCSV)
        button_layout.addWidget(self.load_csv_button)

        # self.add_node_layout = QHBoxLayout()

        self.add_node_button = QPushButton("     Ajouter Noeud", self)
        self.add_node_button.clicked.connect(self.addNode)

        self.node_label_input = QLineEdit(self)
        self.node_label_input.setPlaceholderText("Nom de noeud")

        button_layout.addWidget(self.node_label_input)
        button_layout.addWidget(self.add_node_button)

        self.add_edge_button = QPushButton("      Ajouter Arc", self)
        self.add_edge_button.setIcon(QIcon("icons/arrow.png"))
        self.add_edge_button.clicked.connect(self.addEdge)
        button_layout.addWidget(self.add_edge_button)

        self.add_edge_labels_layout = QHBoxLayout()
        self.edge_source_input = QLineEdit(self)
        self.edge_source_input.setPlaceholderText("Départ")
        self.edge_target_input = QLineEdit(self)
        self.edge_target_input.setPlaceholderText("Arrivé")
        self.edge_wieght_input = QLineEdit(self)
        self.edge_wieght_input.setPlaceholderText("Poids")

        self.add_edge_labels_layout.addWidget(self.edge_source_input)
        self.add_edge_labels_layout.addWidget(self.edge_target_input)
        self.add_edge_labels_layout.addWidget(self.edge_wieght_input)

        button_layout.addLayout(self.add_edge_labels_layout)

        self.remove_node_button = QPushButton("Supprimer Noeud", self)
        self.remove_node_button.clicked.connect(self.removeNode)
        button_layout.addWidget(self.remove_node_button)

        self.remove_node_input = QLineEdit(self)
        self.remove_node_input.setPlaceholderText("Nom de noeud")
        button_layout.addWidget(self.remove_node_input)

        self.remove_edge_layout = QHBoxLayout()

        self.remove_edge_button = QPushButton("Supprimer arc", self)

        self.edge_source_input_remove = QLineEdit(self)
        self.edge_source_input_remove.setPlaceholderText("Départ")
        self.edge_target_input_remove = QLineEdit(self)
        self.edge_target_input_remove.setPlaceholderText("Arrivé")

        self.remove_edge_layout.addWidget(self.edge_source_input_remove)
        self.remove_edge_layout.addWidget(self.edge_target_input_remove)

        button_layout.addWidget(self.remove_edge_button)
        button_layout.addLayout(self.remove_edge_layout)

        self.remove_edge_button.clicked.connect(self.removeEdge)

        self.get_node_information = QPushButton("DES INFORMATIONS SUR LES NOEUDS", self)
        self.get_node_information.clicked.connect(self.nodeInformations)
        button_layout.addWidget(self.get_node_information)

        self.get_graph_information = QPushButton("DES INFORMATIONS SUR LE GRAPHE", self)
        self.get_graph_information.clicked.connect(self.graphInformation)
        button_layout.addWidget(self.get_graph_information)

        self.articulation_points = QPushButton("Point D'Articulation", self)
        button_layout.addWidget(self.articulation_points)
        self.articulation_points.clicked.connect(self.find_articulation_points)

        self.ssc = QPushButton("LES COMPOSANTES CONNEXES", self)
        button_layout.addWidget(self.ssc)
        self.ssc.clicked.connect(self.tarjan_strongly_connected_components)

        self.eul = QPushButton("CHEMIN OU PATH EULERIENNE ", self)
        button_layout.addWidget(self.eul)
        self.eul.clicked.connect(self.find_eulerian_path_or_cycle)

        self.ham = QPushButton("CHEMIN OU PATH HAMILTONIENNE", self)
        button_layout.addWidget(self.ham)
        self.ham.clicked.connect(self.hamiltionian_cycle)

        self.adjcency_matrix_button = QPushButton("LES MATRICES", self)
        button_layout.addWidget(self.adjcency_matrix_button)
        self.adjcency_matrix_button.clicked.connect(
            self.adjacency_matrix_and_incidence_matrix
        )

        self.minimum_spanning_tree_button = QPushButton("ARBRE COUVRANT", self)
        button_layout.addWidget(self.minimum_spanning_tree_button)
        self.minimum_spanning_tree_button.clicked.connect(self.minimum_spanning_tree)

        self.graph_coloring = QPushButton("COLORATION GRAPHE", self)
        button_layout.addWidget(self.graph_coloring)
        self.graph_coloring.clicked.connect(self.coloring_graph)

        # DJIKSTRA CHAFAA
        self.Djikstra = QHBoxLayout()

        self.Djikstra_btn = QPushButton("Djikstra", self)

        self.Djikstra_src = QLineEdit(self)
        self.Djikstra_src.setPlaceholderText("Départ")
        self.Djikstra_dst = QLineEdit(self)
        self.Djikstra_dst.setPlaceholderText("Arrivé")

        self.Djikstra.addWidget(self.Djikstra_src)
        self.Djikstra.addWidget(self.Djikstra_dst)

        button_layout.addWidget(self.Djikstra_btn)
        button_layout.addLayout(self.Djikstra)
        self.Djikstra_btn.clicked.connect(self.runDijkstra)

        # BELLMAN MOUSSA
        self.BellMan_button = QPushButton("      Algorithme BellMan", self)
        self.BellMan_button.clicked.connect(self.bellman)
        button_layout.addWidget(self.BellMan_button)

        self.bellman_layout = QHBoxLayout()

        self.node_source_bl_input = QLineEdit(self)
        self.node_source_bl_input.setPlaceholderText("Nom de noeud source")
        self.bellman_layout.addWidget(self.node_source_bl_input)

        self.node_destination_bl_input = QLineEdit(self)
        self.node_destination_bl_input.setPlaceholderText("Nom de noeud destination")
        self.bellman_layout.addWidget(self.node_destination_bl_input)

        button_layout.addLayout(self.bellman_layout)

        # FORD DHIAA
        self.Ford_button = QPushButton("      Algorithme Ford", self)
        self.Ford_button.clicked.connect(self.Ford)
        button_layout.addWidget(self.Ford_button)

        self.Ford_button_labels_layout = QHBoxLayout()
        self.node_source_input = QLineEdit(self)
        self.node_source_input.setPlaceholderText(" source")
        self.node_destination_input = QLineEdit(self)
        self.node_destination_input.setPlaceholderText("destination")

        self.Ford_button_labels_layout.addWidget(self.node_source_input)
        self.Ford_button_labels_layout.addWidget(self.node_destination_input)
        button_layout.addLayout(self.Ford_button_labels_layout)

        ### Planarité
        self.detect_planarity_button = QPushButton("Détecter Planarité", self)
        self.detect_planarity_button.clicked.connect(self.detectPlanarityy)
        button_layout.addWidget(self.detect_planarity_button)
        ### COUPLAGE
        self.maximal_matching_button = QPushButton("Couplage", self)
        self.maximal_matching_button.clicked.connect(self.findMaximalMatching)
        button_layout.addWidget(self.maximal_matching_button)

        ### fermeture transitive
        self.Transitve_button = QPushButton("      La fermuture Transitive", self)
        self.Transitve_button.clicked.connect(self.fermeture_transitive)
        button_layout.addWidget(self.Transitve_button)

        ### chemin d'ordre n
        self.chemin_longeur_button = QPushButton("    chemine de longeur", self)
        self.chemin_longeur_button.clicked.connect(self.puissance_matrice)
        button_layout.addWidget(self.chemin_longeur_button)

        self.puissance_input = QLineEdit(self)
        self.puissance_input.setPlaceholderText("donne la longeur de chemin")
        button_layout.addWidget(self.puissance_input)

        global html_widget
        html_widget = QWidget()
        global html_layout
        html_layout = QVBoxLayout(html_widget)

        self.graph_frame = QFrame(self)
        html_layout.addWidget(self.graph_frame)

        main_layout.addWidget(button_widget, 1)
        main_layout.addWidget(html_widget, 9)

        self.net = None
        self.directed_bool = False
        self.G = None
        ###export#####
        self.export_csv_button = QPushButton("Exporter vers CSV", self)
        self.export_csv_button.clicked.connect(self.exportGraphToCSV)
        button_layout.addWidget(self.export_csv_button)  # Add the button to the layout



    def exportGraphToCSV(self):
        if self.net is not None:
           
            filename, _ = QFileDialog.getSaveFileName(self, 'Exporter le graphe vers un fichier CSV', '', 'CSV Files (*.csv);;All Files (*)')

            if filename:
                with open(filename, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['Source', 'Target', 'Weight'])  

                    for edge in self.G.edges(data=True):
                        source, target, data = edge
                        weight = data.get("Weight", "")  

                        writer.writerow([source, target, weight])

                    for node in self.G.nodes():
                        if not any(node in edge[:2] for edge in self.G.edges()):
                            
                            writer.writerow([node, "", "0"])

                QMessageBox.information(self, "Exportation réussie", "Le graphe a été exporté vers un fichier CSV avec succès.")
        else:
            QMessageBox.warning(self, "Avertissement", "Il faut générer le graphe d'abord.")
    def export_to_csv(graph, filename):
        with open(filename, mode="w", newline="") as file:
            writer = csv.writer(file)

         
            writer.writerow(["Node", "Edge", "Weight"])

            
            for edge in graph.edges(data=True):
                source, target, data = edge
                writer.writerow([source, target, data.get("Weight", "")])
    def displayGraph(self):
        if self.net is not None:
            self.net.save_graph("graph.html")
            self.webview = QWebEngineView(self)
            self.webview.setHtml(open("graph.html").read())

            if self.graph_frame.layout() is not None:
                global new_html_widget
                new_html_widget = QWidget()
                new_html_layout = QVBoxLayout(new_html_widget)

                self.graph_frame = QFrame(self)
                new_html_layout.addWidget(self.graph_frame)

                graph_layout = QVBoxLayout(self.graph_frame)
                graph_layout.addWidget(self.webview)
                self.graph_frame.setLayout(graph_layout)

                main_layout.addWidget(new_html_widget, 8)

                main_layout.itemAt(1).widget().deleteLater()

            else:
                graph_layout = QVBoxLayout(self.graph_frame)
                graph_layout.addWidget(self.webview)
                self.graph_frame.setLayout(graph_layout)
        else:
            QMessageBox.warning(
                self, "Avertissement", "IL FAUT GENERER LE GRAPHE D'ABORD"
            )

    def addNode(self):
        if self.net is not None:
            node_label = self.node_label_input.text()
            if node_label == "":
                QMessageBox.warning(self, "NOM NO VALIDE", f"DONNER UN NOM VALIDE")
            else:
                self.net.add_node(node_label, size=10)  # for visualisation
                self.G.add_node(node_label)

                self.net.save_graph("graph.html")
                self.webview.setHtml(open("graph.html").read())

                self.node_label_input.setText("")
        else:
            QMessageBox.warning(
                self, "Avertissement", "IL FAUT GENERER LE GRAPHE D'ABORD"
            )

    def addEdge(self):
        if self.net is not None:
            source_label = self.edge_source_input.text()
            target_label = self.edge_target_input.text()
            weight_label = self.edge_wieght_input.text()

            try:
                self.net.add_edge(source_label, target_label)
                self.G.add_edge(
                    source_label,
                    target_label,
                    weight=int(weight_label) if weight_label else 0,
                )
            except AssertionError as e:
                QMessageBox.warning(
                    self,
                    "Avertissement",
                    f"LE NOEUDE {str(str(e).split(' ')[-1])} N'EXISTE PAS",
                )
            self.net.save_graph("graph.html")
            self.webview.setHtml(open("graph.html").read())

            self.edge_source_input.setText("")
            self.edge_target_input.setText("")
        else:
            QMessageBox.warning(
                self, "Avertissement", "IL FAUT GENERER LE GRAPHE D'ABORD"
            )

    def loadCSV(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load CSV File",
            "",
            "CSV Files (*.csv);;All Files (*)",
            options=options,
        )
        if self.directed.isChecked():
            self.directed_bool = True
            self.G = nx.DiGraph()
            print("directed")
        else:
            self.directed_bool = False
            self.G = nx.Graph()
            print("undirected")
        if file_path:
            try:
                df = pd.read_csv(file_path, na_values=["", " "])
                if (
                    "Source" in df.columns
                    and "Target" in df.columns
                    and "Weight" in df.columns
                ):
                    self.G = nx.from_pandas_edgelist(
                        df.dropna(),
                        source="Source",
                        target="Target",
                        edge_attr=["Weight"],
                        create_using=self.G,
                    )
                    isolated_nodes = (
                        df[df["Target"].isna()]["Source"].to_list()
                        + df[df["Source"].isna()]["Target"].to_list()
                    )
                    for node in isolated_nodes:
                        print(node)
                        self.G.add_node(node)

                    edge_attributes = {
                        (source, target): {"title": f"Weight: {weight}"}
                        for source, target, weight in self.G.edges(data="Weight")
                    }
                    nx.set_edge_attributes(self.G, edge_attributes)
                    self.net = Network(
                        notebook=True, directed=self.directed_bool, height="920px"
                    )  # Create a new Pyvis Network object
                    self.net.from_nx(self.G)
                    print(self.is_graph_symmetric())
                    print(self.is_graph_antisymmetric())
                    print(self.is_graph_reflexif())
                    print(self.is_graph_complet())

                    self.displayGraph()
                else:
                    QMessageBox.warning(
                        self,
                        "Avertissement",
                        "LE FICHIER CSV DOIT CONTENIRE LES COLUMNS 'Source' et 'Target'",
                    )
            except Exception as e:
                QMessageBox.critical(self, "Error", f"ERREUR: {str(e)}")

    def removeNode(self):
        node_name = self.remove_node_input.text()

        new_network = Network(
            notebook=True, directed=self.directed_bool, height="920px"
        )

        for node_id in self.net.get_nodes():
            if node_id != node_name:
                new_network.add_node(
                    node_id, label=self.net.get_node(node_id)["label"], size=10
                )

        for edge in self.net.get_edges():
            if node_name not in edge:
                try:
                    new_network.add_edge(edge["from"], edge["to"])
                except:
                    continue

        self.remove_node_input.setText("")
        self.G.remove_node(node_name)
        self.net = new_network
        self.displayGraph()

    def toggle_checkbox(self, checkbox):
        if checkbox == self.directed:
            self.undirected.setChecked(not self.directed.isChecked())

        else:
            self.directed.setChecked(not self.undirected.isChecked())

    def removeEdge(self):
        src = self.edge_source_input_remove.text()
        dst = self.edge_target_input_remove.text()
        removed_edge = list()
        removed_edge.append(src)
        removed_edge.append(dst)

        if not self.directed_bool:
            removed_edge = sorted(removed_edge)

        self.G.remove_edge(src, dst)
        nodes = [node for node in list(self.G.nodes())]
        edges = [edge for edge in self.G.edges() if list(edge) != removed_edge]

        new_network = Network(
            notebook=True, directed=self.directed_bool, height="920px"
        )

        for node_id in nodes:
            new_network.add_node(
                node_id, label=self.net.get_node(node_id)["label"], size=10
            )

        for edge in edges:
            new_network.add_edge(edge[0], edge[1])

        self.net = new_network
        self.displayGraph()

    def nodeInformations(self):
        if self.net is not None:
            self.window2 = QWidget()
            self.window2_layout = QVBoxLayout(self.window2)
            print(self.G.degree())
            for node in self.G.degree():
                node_label_name = QLabel()
                node_label_name.setText(f"LE NOEUDE : {node[0]}")
                self.window2_layout.addWidget(node_label_name)

                node_label_info = QLabel()
                node_label_info.setText(f"LE DEGREE DE NOEUDE : {node[1]}")
                self.window2_layout.addWidget(node_label_info)

                node_label_neighbors = QLabel()
                node_label_neighbors.setText(
                    f"LES VOISINS D'UN NOEUDE : {list(nx.all_neighbors(self.G,node[0]))}"
                )
                self.window2_layout.addWidget(node_label_neighbors)
            self.window2.show()
        else:
            QMessageBox.warning(
                self, "Avertissement", "IL FAUT GENERER LE GRAPHE D'ABORD"
            )

    #### LE GRAPHE CONNEXE ####
    def is_connected(self):
        visited = set()
        self.dfs_is_connected(list(self.G.nodes())[0], visited)
        return len(visited) == len(list(self.G.nodes()))

    def dfs_is_connected(self, node, visited):
        visited.add(node)
        for neighbor in list(self.G[node]):
            if neighbor not in visited:
                self.dfs_is_connected(neighbor, visited)

    ##################### TROUVER LES POINTS D'ARTICULATION ############################
    def find_articulation_points(self):
        if self.net is not None:
            if not self.directed_bool:
                articulation_points = set()

                def dfs(node, parent, discovery_time, low):
                    nonlocal time
                    children = 0
                    discovery_time[node] = time
                    low[node] = time
                    time += 1

                    for neighbor in self.G[node]:
                        if discovery_time[neighbor] == -1:
                            children += 1
                            dfs(neighbor, node, discovery_time, low)
                            low[node] = min(low[node], low[neighbor])

                            if low[neighbor] >= discovery_time[node]:
                                if parent is not None or children > 1:
                                    articulation_points.add(node)

                        elif neighbor != parent:
                            low[node] = min(low[node], discovery_time[neighbor])

                discovery_time = {node: -1 for node in self.G.nodes()}
                low = {node: -1 for node in self.G.nodes()}
                time = 0

                for node in self.G.nodes():
                    if discovery_time[node] == -1:
                        dfs(node, None, discovery_time, low)

                if (len(articulation_points)) > 0:
                    new_network = Network(
                        notebook=True, directed=self.directed_bool, height="920px"
                    )

                    for node in self.G.nodes():
                        color = (
                            random.randint(0, 255),
                            random.randint(0, 255),
                            random.randint(0, 255),
                        )
                        hex_color = "#{:02X}{:02X}{:02X}".format(*color)
                        if node in articulation_points:
                            new_network.add_node(
                                node,
                                label=self.net.get_node(node)["label"],
                                size=10,
                                color=hex_color,
                            )
                        else:
                            new_network.add_node(
                                node, label=self.net.get_node(node)["label"], size=10
                            )

                    for edge in self.net.get_edges():
                        try:
                            new_network.add_edge(edge["from"], edge["to"])
                        except:
                            continue

                    new_network.save_graph("ap.html")
                    self.window4 = QWidget()

                    self.window4_layout = QVBoxLayout(self.window4)
                    self.webviewcc = QWebEngineView()
                    self.webviewcc.setHtml(open("ap.html").read())
                    self.window4_layout.addWidget(self.webviewcc, 95)
                    self.window4.show()
                else:
                    QMessageBox.information(
                        self,
                        "IL N'Y A PAS DES POINTS D'ARTICULATION",
                        "LES POINTS D'ARTICULATION N'EXISTE PAS",
                    )
            else:
                QMessageBox.warning(
                    self, "Avertissement", "LE GRAPHE DOIT ETRE NON ORIENTEE"
                )
        else:
            QMessageBox.warning(
                self, "Avertissement", "IL FAUT GENERER LE GRAPHE D'ABORD"
            )

    #######################################################################################################
    ################# Trouver les composantes connexes ####################################################

    def tarjan_strongly_connected_components(self):
        if self.net is not None:
            index_counter = [0]
            stack = []
            low_link_values = {}
            index_values = {}
            ssc = []

            def strongconnect(node):
                index_values[node] = index_counter[0]
                low_link_values[node] = index_counter[0]
                index_counter[0] += 1
                stack.append(node)

                for successor in self.G[node]:
                    if successor not in index_values:
                        strongconnect(successor)
                        low_link_values[node] = min(
                            low_link_values[node], low_link_values[successor]
                        )
                    elif successor in stack:
                        low_link_values[node] = min(
                            low_link_values[node], index_values[successor]
                        )

                if low_link_values[node] == index_values[node]:
                    connected_component = []
                    while True:
                        successor = stack.pop()
                        connected_component.append(successor)
                        if successor == node:
                            break
                    ssc.append(connected_component)

            for node in self.G.nodes():
                if node not in index_values:
                    strongconnect(node)

            new_network = Network(
                notebook=True, directed=self.directed_bool, height="920px"
            )

            for sc in ssc:
                color = (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255),
                )
                hex_color = "#{:02X}{:02X}{:02X}".format(*color)
                for node in sc:
                    new_network.add_node(
                        node,
                        label=self.net.get_node(node)["label"],
                        size=10,
                        color=hex_color,
                    )

            for edge in self.net.get_edges():
                try:
                    new_network.add_edge(edge["from"], edge["to"])
                except:
                    continue

            new_network.save_graph("ssc.html")
            self.window5 = QWidget()
            self.window5_layout = QVBoxLayout(self.window5)
            self.webviewcc = QWebEngineView()
            self.webviewcc.setHtml(open("ssc.html").read())
            self.window5_layout.addWidget(self.webviewcc, 95)
            if not self.directed_bool:
                msg = QLabel()
                if len(ssc) == 1:
                    s = "Le Graphe Est Connexe"
                else:
                    s = "The Graph N'Est Pas Connexe"
                msg.setText(s)
                self.window5_layout.addWidget(msg, 5)
            self.window5.show()
        else:
            QMessageBox.warning(
                self, "Avertissement", "IL FAUT GENERER LE GRAPHE D'ABORD"
            )

    #######################################################################################################
    ####################### trouver chaine ou cycle eulérienne ################################################
    def find_eulerian_path_or_cycle(self):
        if self.net is not None:
            if not self.directed_bool:
                odd_degree_nodes = [
                    node for node, degree in self.G.degree() if degree % 2 == 1
                ]
                if len(odd_degree_nodes) not in [0, 2]:
                    QMessageBox.information(
                        self,
                        "CHEMIN OU CYCLE N'EXISTE PAS",
                        "IL N'Y A PAS UN CHEMIN OU CYCLE EULERIENNE",
                    )
                else:
                    eulerian_path_or_cycle = []

                    current_node = (
                        odd_degree_nodes[0]
                        if odd_degree_nodes
                        else list(self.G.nodes())[0]
                    )

                    while len(eulerian_path_or_cycle) < self.G.number_of_edges():
                        neighbors = list(self.G.neighbors(current_node))

                        for neighbor in neighbors:
                            if (
                                current_node,
                                neighbor,
                            ) not in eulerian_path_or_cycle and (
                                neighbor,
                                current_node,
                            ) not in eulerian_path_or_cycle:
                                eulerian_path_or_cycle.append((current_node, neighbor))
                                current_node = neighbor
                                break

                    s = eulerian_path_or_cycle[0][0]
                    for i in range(1, len(eulerian_path_or_cycle)):
                        s = s + " --> " + eulerian_path_or_cycle[i][0]
                    s = s + " --> " + eulerian_path_or_cycle[-1][1]

                    if len(odd_degree_nodes) == 0:
                        QMessageBox.information(
                            self, "CYCLE EULERIENNE", "CYCLE EULERIENNE : " + s
                        )
                    else:
                        QMessageBox.information(
                            self, "CHAINE EULERIENNE", "CHAINE EULERIENNE : " + s
                        )
            else:
                QMessageBox.warning(
                    self, "Avertissement", "LE GRAPHE DOIT ETRE NON ORIENTEE"
                )
        else:
            QMessageBox.warning(
                self, "Avertissement", "IL FAUT GENERER LE GRAPHE D'ABORD"
            )

    #########################################################################################
    ############### find hamiltonian cycle ##################################################
    def hamiltionian_cycle(self):
        if self.net is not None:
            if not self.directed_bool:

                def backtrack(path):
                    if len(path) == num_nodes:
                        return path

                    current_node = path[-1]
                    for neighbor in self.G.neighbors(current_node):
                        if neighbor not in path:
                            result = backtrack(path + [neighbor])
                            if result:
                                return result

                    return None

                num_nodes = len(self.G.nodes)
                found_hamiltonian_path = False

                for start_node in self.G.nodes:
                    hamiltonian_path = backtrack([start_node])
                    if hamiltonian_path:
                        found_hamiltonian_path = True
                        break

                if found_hamiltonian_path:
                    s = hamiltonian_path[0]
                    for i in range(1, len(hamiltonian_path)):
                        s = s + " --> " + hamiltonian_path[i]
                    if hamiltonian_path[0] in list(
                        self.G.neighbors(hamiltonian_path[-1])
                    ):
                        QMessageBox.information(
                            self,
                            "Circuit Hamiltionianne",
                            "Circuit Hamiltionianne : " + s,
                        )
                    else:
                        QMessageBox.information(
                            self, "Chaine Hamiltonianne ", "Chaine Hamiltonianne: " + s
                        )
                else:
                    QMessageBox.warning(
                        self, "Avertissmenet ", "Y A PAS UN CYCLE HAMILTIONIANNE"
                    )
            else:
                QMessageBox.warning(
                    self, "Avertissement", "LE GRAPHE DOIT ETRE NON ORIENTEE"
                )
        else:
            QMessageBox.warning(
                self, "Avertissement", "IL FAUT GENERER LE GRAPHE D'ABORD"
            )

    ######################################################################################################
    #################### MATRICE D'ADJACENCE ET D'INCIDENCE ##########################################################
    def adjacency_matrix_and_incidence_matrix(self):
        if self.net is not None:
            num_nodes = len(list(self.G.nodes()))
            edges = list(self.G.edges())

            adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
            incidence_matrix = np.zeros((num_nodes, len(edges)), dtype=int)

            for j, edge in enumerate(edges):
                u, v = edge
                u_idx = list(self.G.nodes()).index(u)
                v_idx = list(self.G.nodes()).index(v)
                adjacency_matrix[u_idx, v_idx] = 1
                adjacency_matrix[v_idx, u_idx] = 1
                incidence_matrix[u_idx, j] = -1 if self.directed_bool else 1
                incidence_matrix[v_idx, j] = 1

            output_string = StringIO()

            print("Node", end="\t", file=output_string)
            for node in list(self.G.nodes()):
                print(node, end="\t", file=output_string)

            print(file=output_string)
            for i in range(num_nodes):
                print(list(self.G.nodes())[i], end="\t", file=output_string)
                for j in range(num_nodes):
                    print(adjacency_matrix[i, j], end="\t", file=output_string)
                print(file=output_string)

            result_string1 = output_string.getvalue()
            output_string.close()

            output_string = StringIO()

            print("Node/Edge", end="\t", file=output_string)
            for edge in edges:
                print(edge, end="\t\t", file=output_string)
            print(file=output_string)

            for i in range(num_nodes):
                print(list(self.G.nodes())[i], end="\t\t", file=output_string)
                for j in range(len(edges)):
                    print(incidence_matrix[i, j], end="\t\t", file=output_string)
                print(file=output_string)

            result_string2 = output_string.getvalue()
            output_string.close()

            self.window6 = QWidget()
            self.window6_layout = QHBoxLayout(self.window6)

            adjacency_matrix = QLabel()
            adjacency_matrix.setText(
                f"Matrice D'Adjacence \n{str(result_string1).strip()}"
            )
            self.window6_layout.addWidget(adjacency_matrix)

            incidence_matrix = QLabel()
            incidence_matrix.setText(
                f"Matrice D'Incidence \n{str(result_string2).strip()}"
            )
            self.window6_layout.addWidget(incidence_matrix)

            self.window6.setGeometry(100, 100, 800, 200)
            self.window6.show()
        else:
            QMessageBox.warning(
                self, "Avertissement", "IL FAUT GENERER LE GRAPHE D'ABORD"
            )

    ######################################################################################################
    #################### ARBRE COUVRANT MINIMUM ##########################################################
    def minimum_spanning_tree(self):
        if self.net is not None:
            if not self.directed_bool:
                parent = {node: node for node in self.G.nodes()}
                edges = sorted(self.G.edges(data=True), key=lambda x: x[2]["Weight"])

                def find(node):
                    if parent[node] != node:
                        parent[node] = find(parent[node])
                    return parent[node]

                def union(node1, node2):
                    n1 = find(node1)
                    n2 = find(node2)
                    parent[n1] = n2

                mst_edges = []

                for edge in edges:
                    source, dest, weight = edge
                    if find(source) != find(dest):
                        mst_edges.append(edge)
                        union(source, dest)

                new_network = Network(
                    notebook=True, directed=self.directed_bool, height="920px"
                )
                for node in self.net.get_nodes():
                    new_network.add_node(node, size=10)

                for edge in self.G.edges(data=True):
                    try:
                        src, dest, info = edge
                        weight = info["Weight"]
                        if edge in mst_edges:
                            new_network.add_edge(
                                src,
                                dest,
                                color="#ff0000",
                                width=3,
                                title=f"Weight: {weight}",
                            )
                        else:
                            new_network.add_edge(src, dest, title=f"Weight: {weight}")
                    except:
                        continue

                new_network.save_graph("mst.html")
                self.window7 = QWidget()

                self.window7_layout = QVBoxLayout(self.window7)
                self.webviewcc = QWebEngineView()
                self.webviewcc.setHtml(open("mst.html").read())
                self.window7_layout.addWidget(self.webviewcc, 95)
                self.window7.show()
            else:
                QMessageBox.warning(
                    self, "Avertissement", "LE GRAPHE DOIT ETRE NON ORIENTEE"
                )
        else:
            QMessageBox.warning(
                self, "Avertissement", "IL FAUT GENERER LE GRAPHE D'ABORD"
            )

    ######################################################################################################
    #################### COLORATION DE GRAPHE ###################################################################
    def coloring_graph(self):
        if self.net is not None:
            if not self.directed_bool:
                nodes = sorted(
                    self.G.nodes(), key=lambda x: self.G.degree(x), reverse=True
                )
                colors = {}

                for node in nodes:
                    neighbor_colors = set(
                        colors.get(neighbor, None)
                        for neighbor in self.G.neighbors(node)
                    )
                    print("neighbor_colors", neighbor_colors)
                    current_color = 1
                    while current_color in neighbor_colors:
                        current_color += 1
                    colors[node] = current_color

                color_groups = {}
                for node, color in colors.items():
                    if color not in color_groups:
                        color_groups[color] = []
                    color_groups[color].append(node)

                new_network = Network(
                    notebook=True, directed=self.directed_bool, height="920px"
                )

                self.window8 = QWidget()
                self.window8_layout = QVBoxLayout(self.window8)

                for color_index, nodes in color_groups.items():
                    label = QLabel()
                    color = (
                        random.randint(0, 255),
                        random.randint(0, 255),
                        random.randint(0, 255),
                    )
                    hex_color = "#{:02X}{:02X}{:02X}".format(*color)
                    label.setText(
                        f"Color {color_index} : {hex_color} \n Nodes : {nodes}"
                    )
                    self.window8_layout.addWidget(label, 1)
                    for node in nodes:
                        new_network.add_node(
                            node,
                            label=self.net.get_node(node)["label"],
                            size=10,
                            color=hex_color,
                        )

                for edge in self.net.get_edges():
                    try:
                        new_network.add_edge(edge["from"], edge["to"])
                    except:
                        continue

                new_network.save_graph("gc.html")

                self.webviewcc = QWebEngineView()
                self.webviewcc.setHtml(open("gc.html").read())
                self.window8_layout.addWidget(self.webviewcc, 99)
                self.window8.show()
            else:
                QMessageBox.warning(
                    self, "Avertissement", "LE GRAPHE DOIT ETRE NON ORIENTEE"
                )
        else:
            QMessageBox.warning(
                self, "Avertissement", "IL FAUT GENERER LE GRAPHE D'ABORD"
            )

    #######################################################################################################
    ############################### INFORMATION SUR LES GRAPHES #############################################
    def is_graph_symmetric(self):
        for node in self.G.nodes():
            for neighbor in self.G.neighbors(node):
                if not self.G.has_edge(neighbor, node):
                    return False
        return True

    def is_graph_antisymmetric(self):
        for node in self.G.nodes():
            for neighbor in self.G.neighbors(node):
                if self.G.has_edge(neighbor, node):
                    return False
        return True

    def is_graph_reflexif(self):
        for node in self.G.nodes():
            if not self.G.has_edge(node, node):
                return False
        return True

    def is_graph_complet(self):
        if self.directed_bool:
            return len(self.G.edges()) == (len(self.G.nodes())) * (
                (len(self.G.nodes())) - 1
            )
        else:
            return (
                len(self.G.edges())
                == ((len(self.G.nodes())) * ((len(self.G.nodes())) - 1)) // 2
            )

    def graphInformation(self):
        if self.net is not None:
            self.window9 = QWidget()
            self.window9_layout = QVBoxLayout(self.window9)
            # symmetric
            symmetric = QLabel()
            symmetric.setText(f"Symmetrique : {self.is_graph_symmetric()}")
            self.window9_layout.addWidget(symmetric)
            # antisymmetric
            antisymmetric = QLabel()
            antisymmetric.setText(f"Antisymmetrique : {self.is_graph_antisymmetric()}")
            self.window9_layout.addWidget(antisymmetric)
            # reflexif
            reflexif = QLabel()
            reflexif.setText(f"Reflxif : {self.is_graph_reflexif()}")
            self.window9_layout.addWidget(reflexif)
            # complet
            complet = QLabel()
            complet.setText(f"Complet : {self.is_graph_complet()}")
            self.window9_layout.addWidget(complet)

            self.window9.show()
        else:
            QMessageBox.warning(
                self, "Avertissement", "IL FAUT GENERER LE GRAPHE D'ABORD"
            )

    ########################################################################################################
    ############################### DJIKSTRA ###########################################################
    def dijkstra_algorithm(self, start_node, end_node):
        if self.G is not None:
            # verifier si depart et arret sont dans le graphe
            if start_node not in self.G.nodes or end_node not in self.G.nodes:
                QMessageBox.warning(
                    self,
                    "Avertissement",
                    f"Les nœuds de départ ou d'arrivée spécifiés n'existent pas dans le graphe.",
                )
                return None

            # verifier si le poid negative
            for edge in self.G.edges(data=True):
                weight = edge[2].get("Weight", 0)
                if weight < 0:
                    QMessageBox.warning(
                        self,
                        "Avertissement",
                        "L'algorithme de Dijkstra ne peut pas gérer les graphes avec des poids négatifs.",
                    )
                    return None
            # initialisation des distances et de la file
            distances = {node: float("infinity") for node in self.G.nodes}
            distances[start_node] = 0
            priority_queue = [(0, start_node)]
            predecessors = {}

            while priority_queue:
                # defiler le noeud qui a la petite distance dans la file
                current_distance, current_node = heappop(priority_queue)
                # iteré les voisin
                for neighbor, weight in self.G[current_node].items():
                    if weight["Weight"] < 0:
                        QMessageBox.warning(
                            self,
                            "Avertissement",
                            "L'algorithme de Dijkstra ne peut pas gérer les graphes avec des poids négatifs.",
                        )
                        return None
                    # calcul de distance entre noeud et son successeur
                    distance = distances[current_node] + weight["Weight"]
                    # mise a jour de la distance
                    if distance < distances[neighbor]:
                        distances[neighbor] = distance
                        predecessors[neighbor] = current_node
                        heappush(priority_queue, (distance, neighbor))

            # construction de chemin
            path = []
            current = end_node
            while current is not None:
                path.insert(0, current)
                current = predecessors.get(current)
            # verifie s`il existe un chemin
            if distances[end_node] == float("infinity"):
                QMessageBox.warning(self, "Avertissement", "Aucun chemin trouvé")
                # retourner le chemin
                return None

            return path
        else:
            QMessageBox.warning(
                self, "Avertissement", "IL FAUT GENERER LE GRAPHE D'ABORD"
            )

    def cycle(self):
        visited = set()
        stack = set()

        def dfs(node):
            visited.add(node)
            stack.add(node)

            for neighbor in self.G[node]:
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in stack:
                    return True

            stack.remove(node)
            return False

        for node in self.G.nodes:
            if node not in visited:
                if dfs(node):
                    return True

        return False

    def runDijkstra(self):
        start_node = self.Djikstra_src.text()
        end_node = self.Djikstra_dst.text()

        if self.directed_bool:
            if start_node and end_node:
                self.shortest_path = self.dijkstra_algorithm(start_node, end_node)
                if self.shortest_path is not None:
                    self.colorShortestPath(self.shortest_path)
            else:
                QMessageBox.warning(
                    self, "Avertissement", "Entrez le node de départ et d'arrivé"
                )
        else:
            QMessageBox.warning(self, "Avertissement", "LE GRAPHE DOIT ETRE ORIENTEE")

    def colorShortestPath(self, shortest_path):
        if self.net is not None:
            # instancier un reseau
            new_network = Network(
                notebook=True, directed=self.directed_bool, height="920px"
            )
            # met tous les noeuds dans le reseau
            for node in self.net.get_nodes():
                new_network.add_node(node, size=10)

            edges = self.G.edges(data=True)

            for edge in edges:
                try:
                    src, dest, info = edge
                    weight = info["Weight"]
                    # verifier si le l`arc appartient au chemin
                    if (src, dest) in zip(shortest_path, shortest_path[1:]):
                        # ajouter l`arc au reseau et le coloré
                        new_network.add_edge(
                            src,
                            dest,
                            color="#ff0000",
                            width=3,
                            title=f"Weight: {weight}",
                        )
                    else:
                        # ajouter l`arc sans colore car n`appartient pas au chemin
                        new_network.add_edge(src, dest, title=f"Weight: {weight}")
                except Exception as e:
                    print(f"An error occurred while processing edge: {str(e)}")
            # grader le graphe dans un fichier html
            new_network.save_graph("shortest_path.html")
            # creer la fenetre + affichage
            self.window7 = QWidget()
            self.window7_layout = QVBoxLayout(self.window7)
            self.webviewcc = QWebEngineView()
            self.webviewcc.setHtml(open("shortest_path.html").read())
            self.window7_layout.addWidget(self.webviewcc, 95)
            self.window7.show()
        else:
            QMessageBox.warning(self, "Avertissement", "Generate the graph first.")

    ##############################################################################################
    ############################# PLANARITE #####################################################
    def detectPlanarityy(self):
        if self.net is not None:
            if not self.directed_bool:
                if self.G:
                    is_planar, embedding = nx.check_planarity(self.G)
                    result_text = (
                        "Le graphe est planaire."
                        if is_planar
                        else "Le graphe n'est pas planaire."
                    )
                    QMessageBox.information(self, "Result", result_text)
                    if is_planar:
                        self.showPlanarEmbedding(embedding)
                else:
                    QMessageBox.information(self, "Result", result_text)
            else:
                QMessageBox.warning(
                    self, "Avertissement", "LE GRAPHE DOIT ETRE NON ORIENTEE"
                )
        else:
            QMessageBox.warning(
                self, "Avertissement", "IL FAUT GENERER LE GRAPHE D'ABORD"
            )

    def showPlanarEmbedding(self, embedding):
        pos = nx.combinatorial_embedding_to_pos(embedding)
        nx.draw(
            self.G,
            pos,
            with_labels=True,
            font_weight="bold",
            node_size=500,
            node_color="skyblue",
            font_color="black",
            font_size=10,
        )
        plt.title("Planar Embedding")
        plt.show()

    ########################################################################################
    ############################# COUPLAGE ################################################
    def findMaximalMatching(self):
        if self.net is not None:
            if not self.directed_bool:
                # Assuming self.G is your graph
                matching = nx.maximal_matching(self.G)

                new_network = Network(
                    notebook=True, directed=self.directed_bool, height="920px"
                )

                for node in self.net.get_nodes():
                    new_network.add_node(node, size=10)

                edges = self.G.edges(data=True)

                for edge in edges:
                    try:
                        src, dest, info = edge
                        weight = info.get(
                            "Weight", 1
                        )  # Default weight to 1 if not specified
                        if (src, dest) in matching or (dest, src) in matching:
                            new_network.add_edge(
                                src,
                                dest,
                                color="#00ff00",
                                width=3,
                                title=f"Weight: {weight} (Matching)",
                            )
                        else:
                            new_network.add_edge(src, dest, title=f"Weight: {weight}")
                    except Exception as e:
                        print(f"An error occurred while processing edge: {str(e)}")

                new_network.save_graph("max_matching.html")
                self.window7 = QWidget()
                self.window7_layout = QVBoxLayout(self.window7)
                self.webviewcc = QWebEngineView()
                self.webviewcc.setHtml(open("max_matching.html").read())
                self.window7_layout.addWidget(self.webviewcc, 95)
                self.window7.show()
            else:
                QMessageBox.warning(
                    self, "Avertissement", "LE GRAPHE DOIT ETRE NON ORIENTEE"
                )
        else:
            QMessageBox.warning(self, "Avertissement", "Generate the graph first.")

    #######################################################################################
    ################## FORD ###############################################################

    def Ford(self):
        if self.net is not None:
            # Vérifier si le graphe est orienté
            if not self.directed_bool:
                QMessageBox.warning(
                    self,
                    "Graph Non-Orienté",
                    "L'algorithme de Ford ne s'applique qu'aux graphes orientés.",
                )
                return

            node_source = self.node_source_input.text()
            print(node_source)
            node_destination = self.node_destination_input.text()
            # Vérifier si le nœud source existe
            if node_source not in self.G.nodes():
                QMessageBox.warning(
                    self,
                    "Nœud Source Invalide",
                    f"Le nœud source '{node_source}' n'existe pas dans le graphe.",
                )
                return

            # Initialiser les distances avec l'infini
            distances = {node: float("inf") for node in self.G.nodes}
            predecessors = {node: None for node in self.G.nodes}
            distances[node_source] = 0

            # Implémentation de l'algorithme de Ford
            for _ in range(len(self.G.nodes()) - 1):
                for edge in self.G.edges(data=True):
                    source, target, weight = edge[0], edge[1], edge[2].get("Weight", 1)
                    if distances[source] + weight < distances[target]:
                        distances[target] = distances[source] + weight
                        predecessors[target] = source
            # Check for absorbing cycles after n-1 iterations

            absorbing_cycle = self.detect_absorbing_cycle(distances)

            if absorbing_cycle:
                # There is an absorbing cycle
                QMessageBox.warning(
                    self, "Avertissement", "LE GRAPHE CONTIENT UN CYCLE ABSORBANT."
                )

                return

            path = self.find_path_ford(predecessors, node_destination)

            if path:
                print(path)

                print("Distance totale : {}".format(distances[node_destination]))
                new_network = Network(
                    notebook=True, directed=self.directed_bool, height="920px"
                )
                for node in self.net.get_nodes():
                    new_network.add_node(node, size=10)

                for edge in self.G.edges(data=False):
                    try:
                        src, dest = edge

                        if edge in path:
                            new_network.add_edge(src, dest, color="#ff0000", width=3)
                        else:
                            new_network.add_edge(src, dest)
                    except:
                        continue

                new_network.save_graph("mst.html")
                self.window7 = QWidget()
                self.window7_layout = QVBoxLayout(self.window7)
                self.webviewcc = QWebEngineView()
                self.webviewcc.setHtml(open("mst.html").read())
                self.window7_layout.addWidget(self.webviewcc, 95)
                self.window7.show()
            else:
                QMessageBox.information(self, "Avertissement", "LE CHEMIN N'EXIST PAS")
        else:
            QMessageBox.warning(
                self, "Avertissement", "IL FAUT GENERER LE GRAPHE D'ABORD"
            )

    def detect_absorbing_cycle(self, distances):
        for edge in self.G.edges(data=True):
            source, target, weight = edge[0], edge[1], edge[2].get("Weight", 1)
            if distances[source] + weight < distances[target]:
                return True
        return False

    def find_path_ford(self, predecessors, node_destination):
        path = []
        path_edges = []
        visited = set()

        while node_destination and node_destination not in visited:
            visited.add(node_destination)
            path.insert(0, node_destination)
            predecessors_node = predecessors.get(node_destination, None)

            if predecessors_node is not None:
                edge = (predecessors_node, node_destination)
                path_edges.insert(0, edge)
            node_destination = predecessors_node

        if node_destination in visited:
            absorbing_cycle = []
            absorbing_cycle_found = False

            # Reconstruct the absorbing cycle
            for node in path:
                if node == node_destination:
                    absorbing_cycle_found = True
                if absorbing_cycle_found:
                    absorbing_cycle.append(node)

            print("Absorbing Cycle:", absorbing_cycle)

        return path_edges

    #######################################################################################
    ################################## BELLMAN ############################################

    def bellman(self):
        if self.G is not None:
            if self.directed_bool:
                if not self.cycle():
                    if self.directed_bool:
                        print(self.G.is_directed)
                        source_node = self.node_source_bl_input.text()
                        destination_node = self.node_destination_bl_input.text()
                        if (
                            source_node not in self.G.nodes
                            or destination_node not in self.G.nodes
                        ):
                            QMessageBox.warning(
                                self,
                                "Avertissement",
                                "Les nœuds source ou destination n'existent pas dans le graphe.",
                            )
                            return
                        else:
                            sequence = self.do_topological_sort()
                            sequence2 = {node: v for node, v in sequence}
                            sorted_sequence = [node[0] for node in sequence]

                            distance = {node: float("inf") for node, _ in sequence}
                            parent = {node: None for node, _ in sequence}

                            distance[source_node] = 0

                            for u in sorted_sequence:
                                for v in sorted_sequence:
                                    if self.G.has_edge(u, v):
                                        weight = self.G.get_edge_data(u, v).get(
                                            "Weight"
                                        )
                                        if distance[v] > distance[u] + weight:
                                            distance[v] = distance[u] + weight
                                            parent[v] = u

                            path = self.find_path(parent, destination_node)
                            if path:
                                new_network = Network(
                                    notebook=True,
                                    directed=self.directed_bool,
                                    height="920px",
                                )
                                for node in self.net.get_nodes():
                                    new_network.add_node(node, size=10)

                                for edge in self.G.edges(data=False):
                                    try:
                                        src, dest = edge

                                        if edge in path:
                                            new_network.add_edge(
                                                src, dest, color="#ff0000", width=3
                                            )
                                        else:
                                            new_network.add_edge(src, dest)
                                    except:
                                        continue

                                new_network.save_graph("mst.html")
                                self.window7 = QWidget()
                                self.window7_layout = QVBoxLayout(self.window7)
                                ####
                                s = "TRI TOPOLOGIQUE : "
                                for el in sequence:
                                    s = s + str(el) + " "
                                label = QLabel()
                                label.setText(s)
                                self.window7_layout.addWidget(label)

                                s = "LES PARENTS : "
                                for el in parent:
                                    s = s + str(el) + " : " + str(parent[el]) + " | "
                                label = QLabel()
                                label.setText(s)
                                self.window7_layout.addWidget(label)

                                s = "LES COUTS : "
                                for el in distance:
                                    s = s + str(el) + " : " + str(distance[el]) + " | "
                                label = QLabel()
                                label.setText(s)
                                self.window7_layout.addWidget(label)

                                ###
                                self.webviewcc = QWebEngineView()
                                self.webviewcc.setHtml(open("mst.html").read())
                                self.window7_layout.addWidget(self.webviewcc, 95)
                                self.window7.show()
                            else:
                                QMessageBox.warning(
                                    self, "Avertissement", "LE PATH N'EXIST PAS"
                                )
                    else:
                        QMessageBox.warning(
                            self, "Avertissement", "Upload un graphe orienté."
                        )
                else:
                    QMessageBox.warning(
                        self, "Avertissement", "Le graphe contient un cycle."
                    )
            else:
                QMessageBox.warning(
                    self, "Avertissement", "LE GRAPHE DOIT ETRE ORIENTE"
                )
        else:
            QMessageBox.warning(
                self, "Avertissement", "IL FAUT GENERER LE GRAPHE D'ABORD"
            )

    def do_topological_sort(self):
        graph = copy.deepcopy(self.G)
        iteration = 1
        sequence = []
        while True:
            nodes_without_predecessors = [
                node
                for node in graph.nodes()
                if len(list(graph.predecessors(node))) == 0
            ]

            if len(nodes_without_predecessors) == 0:
                break

            for node in nodes_without_predecessors:
                sequence.append((node, iteration))
                iteration += 1

            for node in nodes_without_predecessors:
                successors = list(graph.successors(node))

                for successor in successors:
                    graph.remove_edge(node, successor)

                graph.remove_node(node)

        return sequence

    def find_path(self, predecessors, destination_node):
        path = []
        path_edges = []
        while destination_node is not None:
            path.insert(0, destination_node)
            predecessors_node = predecessors.get(destination_node, None)
            edge = (predecessors_node, destination_node)
            destination_node = predecessors_node
            path_edges.insert(0, edge)
        path_edges.pop(0)
        return path_edges

    #######################################################################################
    ########################## fermeture transitive ######################################
    def fermeture_transitive(self):
        if self.net is not None:
            num_nodes = len(list(self.G.nodes()))

            if self.directed_bool:
                edges = list(self.G.edges())
            else:
                edges = list(self.G.edges()) + [(v, u) for u, v in self.G.edges()]

            adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

            for j, edge in enumerate(edges):
                u, v = edge
                u_idx = list(self.G.nodes()).index(u)

                v_idx = list(self.G.nodes()).index(v)

                adjacency_matrix[u_idx, v_idx] = 1
            closure_matrix = adjacency_matrix

            while True:
                previous_closure_matrix = [row.copy() for row in closure_matrix]

                for i in range(num_nodes):
                    for j in range(num_nodes):
                        for k in range(num_nodes):
                            closure_matrix[i][j] = (
                                closure_matrix[i][k] and closure_matrix[k][j]
                            ) or closure_matrix[i][j]

                if all(
                    closure_matrix[i][j] == previous_closure_matrix[i][j]
                    for i in range(len(closure_matrix))
                    for j in range(len(closure_matrix[0]))
                ):
                    break

            output_string = StringIO()
            print("Node", end="\t", file=output_string)
            for node in list(self.G.nodes()):
                print(node, end="\t", file=output_string)
            print(file=output_string)
            for i in range(num_nodes):
                print(list(self.G.nodes())[i], end="\t", file=output_string)
                for j in range(num_nodes):
                    print(adjacency_matrix[i, j], end="\t", file=output_string)
                print(file=output_string)
            result_string1 = output_string.getvalue()
            output_string.close()
            output_string = StringIO()

            closure_matrix2 = "\n".join(
                ["\t".join(map(str, row)) for row in closure_matrix]
            )
            self.window6 = QWidget()
            self.window6_layout = QHBoxLayout(self.window6)

            closure_matrix2 = QLabel()
            closure_matrix2.setText(
                f"Matrice D'Adjacence \n{str(result_string1).strip()}"
            )
            self.window6_layout.addWidget(closure_matrix2)
            self.window6.setGeometry(100, 100, 800, 200)
            self.window6.show()
        else:
            QMessageBox.warning(
                self, "Avertissement", "IL FAUT GENERER LE GRAPHE D'ABORD"
            )

    ######################################################################################
    ############################# chemin d'order n #######################################
    def puissance_matrice(self):
        if self.net is not None:
            try:
                puissance = int(self.puissance_input.text())

                num_nodes = len(list(self.G.nodes()))

                if self.directed_bool:
                    edges = list(self.G.edges())
                else:
                    edges = list(self.G.edges()) + [(v, u) for u, v in self.G.edges()]

                adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

                for j, edge in enumerate(edges):
                    u, v = edge
                    u_idx = list(self.G.nodes()).index(u)
                    v_idx = list(self.G.nodes()).index(v)
                    adjacency_matrix[u_idx, v_idx] = 1

                resultat = adjacency_matrix
                for _ in range(1, puissance):
                    resultat = self.produit_matrice(
                        resultat, adjacency_matrix, num_nodes
                    )

                output_string = StringIO()
                num_nodes = len(list(self.G.nodes()))

                print("Node", end="\t", file=output_string)
                for node in list(self.G.nodes()):
                    print(node, end="\t", file=output_string)

                print(file=output_string)

                for i in range(num_nodes):
                    print(list(self.G.nodes())[i], end="\t", file=output_string)
                    for j in range(num_nodes):
                        print(resultat[i][j], end="\t", file=output_string)
                    print(file=output_string)

                result_string = output_string.getvalue()
                output_string.close()
                self.window6 = QWidget()
                self.window6_layout = QHBoxLayout(self.window6)

                resultat = QLabel()
                resultat.setText(
                    f" LES CHEMINS D'ORDRE {puissance} \n{str(result_string).strip()}"
                )
                self.window6_layout.addWidget(resultat)
                self.window6.setGeometry(100, 100, 800, 200)
                self.window6.show()
            except Exception:
                QMessageBox.warning(
                    self, "Avertissement", "IL FAUT DONNER UN PUISSANCE VALID"
                )
        else:
            QMessageBox.warning(
                self, "Avertissement", "IL FAUT GENERER LE GRAPHE D'ABORD"
            )

    def produit_matrice(self, matrice1, matrice2, num_nodes):
        # Vérifie si les dimensions sont compatibles pour le produit matriciel

        # Initialisation de la matrice résultat avec des zéros
        resultat = [[0 for _ in range(num_nodes)] for _ in range(num_nodes)]

        # Calcul du produit matriciel
        for i in range(num_nodes):
            for j in range(num_nodes):
                for k in range(num_nodes):
                    resultat[i][j] += matrice1[i][k] * matrice2[k][j]

        return resultat

    ######################################################################################


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())