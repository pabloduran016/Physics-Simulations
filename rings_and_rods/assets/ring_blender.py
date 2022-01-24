import bpy

obdata = bpy.context.object.data

print('Vertices')
for vertice in obdata.vertices:
    print(f'[{vertice.co.x}, {vertice.co.y}, {vertice.co.z}],')

#print('Edges')
#for edge in obdata.edges:
#    print(f'[{edge.vertices[0]}, {edge.vertices[1]}],')
#    
#print('Surfaces')
#for face in obdata.polygons:
#    print(f'[{face.vertices[0]}, {face.vertices[1]}, '\
#          f'{face.vertices[2]}, {face.vertices[3]}],')
print('Normals')
for face in obdata.polygons:
    print(f'    [{face.normal[0]}, {face.normal[1]}, {face.normal[1]}],')
