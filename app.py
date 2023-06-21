from qa.question_answering import QuestionAnswering
from qa.data_loader import MetaQADataLoader
import gradio as gr
import networkx as nx
import plotly.graph_objects as go


class QADemo:
    def __init__(self):
        self.data_loader = MetaQADataLoader('./data')
        self.qa = QuestionAnswering('navidmadani/nl2logic_t5small_metaqa', self.data_loader)

    def build_graph(self, query_trace, relation_trace, question_entity):
        G = nx.DiGraph()
        G.add_node(question_entity, color='red', name=question_entity)
        id2hop = {'X': 1, 'Y': 2, 'Z': 3}
        prev_node = question_entity
        for ans in query_trace:
            for id, ent in ans.items():
                G.add_node(ent, color='blue', name=ent)
                G.add_edge(prev_node, ent, label=f'{relation_trace[id2hop[id]-1]}')
                prev_node = ent
            prev_node = question_entity

        return G

    def run(self, question):
        answer_component = self.qa.answer_question(question)
        G = self.build_graph(answer_component['trace'], answer_component['relation_trace'], answer_component['qent'])

        pos = nx.spring_layout(G)

        node_trace = go.Scatter(
            x=[],
            y=[],
            text=[],
            mode='markers',
            hoverinfo='text',
            marker=dict(
                color=[],
                size=15,
                line_width=2
            )
        )

        for node in G.nodes():
            x, y = pos[node]
            node_trace['x'] += tuple([x])
            node_trace['y'] += tuple([y])
            if str(node) in  answer_component['answers']:
                node_trace['marker']['color'] += tuple(['green'])
            elif str(node) == answer_component['qent']:
                node_trace['marker']['color'] += tuple(['orange'])
            else:
                node_trace['marker']['color'] += tuple(['black'])
            node_trace['text'] += tuple([str(node)])  # Display node name

        edge_trace = go.Scatter(
            x=[],
            y=[],
            text=[],
            line=dict(width=1.0, color='#888'),
            hoverinfo='text',
            mode='lines'
        )

        for edge, label in zip(G.edges(), nx.get_edge_attributes(G, 'label').values()):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace['x'] += tuple([x0, x1, None])
            edge_trace['y'] += tuple([y0, y1, None])
            edge_trace['text'] += tuple(label)  # Add edge names

        # Create annotations for edge names
        annotations = []
        for edge, label in zip(G.edges(), nx.get_edge_attributes(G, 'label').values()):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            annotation = go.Annotation(
                x=(x0 + x1) / 2,
                y=(y0 + y1) / 2,
                text=label,
                showarrow=False
            )
            annotations.append(annotation)

        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title='Graph',
                titlefont=dict(size=16),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=annotations,  # Add edge name annotations
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
        )
        return fig, answer_component


qa_demo = QADemo()
demo = gr.Interface(fn=qa_demo.run, inputs="text", outputs=["plot", "text"])
demo.launch()