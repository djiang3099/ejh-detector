# AMME4710 - EJH Detector 2020
# Circuit Digitaliser 

# 470205127
# 470355499
# 470425954

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def circuit_plot(compNames,adj):
# if __name__ == '__main__':

    G = nx.Graph()


    lineNames=[]

    #Give the lines names
    for line_idx in range(len(adj)):
        lineNames.append('l'+str(line_idx))
        print(line_idx)
    lineNames=np.asarray(lineNames)

    #Add nodes from the components and lines
    G.add_nodes_from(compNames)
    G.add_nodes_from(lineNames)

    #Define the connections between components and lines
    for lineIndex in range(len(adj)):
        print(adj[lineIndex,:])
        for compIndex, connectionType in enumerate(adj[lineIndex,:]):
            if connectionType>0:
                #Label the connection type as postive or negative
                if connectionType==1:
                    G.add_edge(compNames[compIndex],lineNames[lineIndex],label='-ve')
                else:
                    G.add_edge(compNames[compIndex],lineNames[lineIndex],label='+ve')




    #Draw the graph
    #How to spread out the graph
    pos=nx.planar_layout(G)
    #nx.draw(G,pos,with_labels=True,node_color='#ffffff')
    nx.draw(G,pos,node_color='#ffffff')
    labels = nx.get_edge_attributes(G,'label')
    nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)


    img=[]
    #ADAPTED FROM https://gist.github.com/shobhit/3236373
    
    ax=plt.gca()
    fig=plt.gcf()
    trans = ax.transData.transform
    trans2 = fig.transFigure.inverted().transform
    imsize = 0.1 # this is the image size
    #Loop through the nodes and overlay and image
    for n in G.nodes():
        (x,y) = pos[n]
        xx,yy = trans((x,y)) # figure coordinates
        xa,ya = trans2((xx,yy)) # axes coordinates
        a = plt.axes([xa-imsize/2.0,ya-imsize/2.0, imsize, imsize ])
        a.imshow(mpimg.imread("comp/"+n[0]+".png"))
        a.set_aspect('equal')
        a.axis('off')

    plt.show()
