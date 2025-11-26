# mini_gencast/mesh.py
import meshzoo
import numpy as np

def build_icosahedral_mesh(refinement_level=2):
    points, cells = meshzoo.icosa_sphere(refinement_level)
    x, y, z = points[:,0], points[:,1], points[:,2]
    lat = np.arcsin(z) * 180/np.pi
    lon = np.arctan2(y, x) * 180/np.pi
    nodes = np.stack([lat, lon], axis=1)
    edges = set()
    for tri in cells:
        i,j,k = tri
        for a,b in [(i,j),(j,k),(k,i)]:
            edges.add(tuple(sorted((a,b))))
    edges = np.array(list(edges))
    return nodes, edges
