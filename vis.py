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

    G = nx.Graph()


    #adj=np.asarray([[2,2,2],
    #[1,1,1]])
    #!!print(adj)
    #compNames=np.asarray(['v','r1','r2'])
    lineNames=[]

    for line_idx in range(len(adj)):
        lineNames.append('l'+str(line_idx))
        print(line_idx)
    lineNames=np.asarray(lineNames)
    '''
    for element in compNames:
        G.add_node(element,image= mpimg.imread("comp/"+element[0]+".png"),size=0.1)'''
    G.add_nodes_from(compNames)
    G.add_nodes_from(lineNames)

    for lineIndex in range(len(adj)):
        print(adj[lineIndex,:])
        for compIndex, connectionType in enumerate(adj[lineIndex,:]):
            if connectionType>0:
                #We have a connection
                #print(compNames[compIndex])
                #could use weight as an alternative
                if connectionType==1:
                    G.add_edge(compNames[compIndex],lineNames[lineIndex],label='-ve')
                else:
                    G.add_edge(compNames[compIndex],lineNames[lineIndex],label='+ve')



    #pos=nx.planar_layout(G)


    #How to spread out the graph
    pos=nx.planar_layout(G)
    nx.draw(G,pos,with_labels=True,node_color='#ffffff')
    #nx.draw(G,pos,with_labels=True)

    labels = nx.get_edge_attributes(G,'label')
    nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)


    img=[]
    #ADAPTED FROM https://gist.github.com/shobhit/3236373
    
    ax=plt.gca()
    fig=plt.gcf()
    trans = ax.transData.transform
    trans2 = fig.transFigure.inverted().transform
    imsize = 0.1 # this is the image size
    for n in G.nodes():
        (x,y) = pos[n]
        xx,yy = trans((x,y)) # figure coordinates
        xa,ya = trans2((xx,yy)) # axes coordinates
        a = plt.axes([xa-imsize/2.0,ya-imsize/2.0, imsize, imsize ])
        a.imshow(mpimg.imread("comp/"+n[0]+".png"))
        a.set_aspect('equal')
        a.axis('off')
    print(compNames)
    print(adj)
    plt.show()
