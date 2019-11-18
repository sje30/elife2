import pandas as pd
import numpy as np
import os
import sys
import plotly
plotly.tools.set_credentials_file(username='unimusic', api_key='I1TfBXlucguG0uXh9lPV')
import plotly.plotly as py
import plotly.graph_objs as go
import random

'''
Plots the nucleus locations in 3D for each dataset along with the EM dataset using Plot.ly.

Author: Angela Zhang
First Written: 10/3/2018
Last Edited: 1/4/2019
'''

plot_process = True
plot_transformed = True
compare_metrics = False

home_folder = '/Users/Unimusic/Desktop/Smith_Lab'
label_folder = os.path.join(home_folder, 'flipped prs 10.3.18')
gt_file = os.path.join(home_folder, 'all cells - formatted for angela.xlsx')
transformed_all = []
num = 0

df1 = pd.read_excel(gt_file)

inds1 = df1.index[df1['ID'].str.lower().str.contains('pr-')]
inds2 = df1.index[df1['ID'].str.lower().str.startswith('ddn')]
inds3 = df1.index[df1['ID'].str.lower().str.startswith('ant')]

inds = inds1.append(inds2)
inds = inds.append(inds3)

names = df1.values[inds, 1]
df1_names = df1.iloc[inds,:]
cell_types = []
for name in names:
	prefix = name[0]
	cell_types.append(prefix)
cell_types = set(cell_types)
orig_coords = []
all_traces = []

layout = go.Layout(showlegend=False,
		scene=dict(
		  xaxis=dict(
				title='',
				showgrid=False,
				zeroline=False,
				showline=False,
				showticklabels=False),
		  yaxis=dict(
				title='',
				showgrid= False,
				zeroline=False,
				showline=False,
				showticklabels=False),
		 zaxis=dict(
				title='',
				showgrid= False,
				zeroline=False,
				showline=False,
				showticklabels=False)),
		scene2=dict(
		  xaxis=dict(
				title='',
				showgrid=False,
				zeroline=False,
				showline=False,
				showticklabels=False),
		  yaxis=dict(
				title='',
				showgrid= False,
				zeroline=False,
				showline=False,
				showticklabels=False),
		 zaxis=dict(
				title='',
				showgrid= False,
				zeroline=False,
				showline=False,
				showticklabels=False)),
		scene3=dict(
		  xaxis=dict(
				title='',
				showgrid=False,
				zeroline=False,
				showline=False,
				showticklabels=False),
		  yaxis=dict(
				title='',
				showgrid= False,
				zeroline=False,
				showline=False,
				showticklabels=False),
		 zaxis=dict(
				title='',
				showgrid= False,
				zeroline=False,
				showline=False,
				showticklabels=False)),
		scene4=dict(
		  xaxis=dict(
				title='',
				showgrid=False,
				zeroline=False,
				showline=False,
				showticklabels=False),
		  yaxis=dict(
				title='',
				showgrid= False,
				zeroline=False,
				showline=False,
				showticklabels=False),
		 zaxis=dict(
				title='',
				showgrid= False,
				zeroline=False,
				showline=False,
				showticklabels=False)),
		scene5=dict(
		  xaxis=dict(
				title='',
				showgrid=False,
				zeroline=False,
				showline=False,
				showticklabels=False),
		  yaxis=dict(
				title='',
				showgrid= False,
				zeroline=False,
				showline=False,
				showticklabels=False),
		 zaxis=dict(
				title='',
				showgrid= False,
				zeroline=False,
				showline=False,
				showticklabels=False)),
		scene6=dict(
		  xaxis=dict(
				title='',
				showgrid=False,
				zeroline=False,
				showline=False,
				showticklabels=False),
		  yaxis=dict(
				title='',
				showgrid= False,
				zeroline=False,
				showline=False,
				showticklabels=False),
		 zaxis=dict(
				title='',
				showgrid= False,
				zeroline=False,
				showline=False,
				showticklabels=False)),
		scene7=dict(
		  xaxis=dict(
				title='',
				showgrid=False,
				zeroline=False,
				showline=False,
				showticklabels=False),
		  yaxis=dict(
				title='',
				showgrid= False,
				zeroline=False,
				showline=False,
				showticklabels=False),
		 zaxis=dict(
				title='',
				showgrid= False,
				zeroline=False,
				showline=False,
				showticklabels=False)),
		scene8=dict(
		  xaxis=dict(
				title='',
				showgrid=False,
				zeroline=False,
				showline=False,
				showticklabels=False),
		  yaxis=dict(
				title='',
				showgrid= False,
				zeroline=False,
				showline=False,
				showticklabels=False),
		 zaxis=dict(
				title='',
				showgrid= False,
				zeroline=False,
				showline=False,
				showticklabels=False)),
		scene9=dict(
		  xaxis=dict(
				title='',
				showgrid=False,
				zeroline=False,
				showline=False,
				showticklabels=False),
		  yaxis=dict(
				title='',
				showgrid= False,
				zeroline=False,
				showline=False,
				showticklabels=False),
		 zaxis=dict(
				title='',
				showgrid= False,
				zeroline=False,
				showline=False,
				showticklabels=False)),
		scene10=dict(
		  xaxis=dict(
				title='',
				showgrid=False,
				zeroline=False,
				showline=False,
				showticklabels=False),
		  yaxis=dict(
				title='',
				showgrid= False,
				zeroline=False,
				showline=False,
				showticklabels=False),
		 zaxis=dict(
				title='',
				showgrid= False,
				zeroline=False,
				showline=False,
				showticklabels=False))
		)

fig = plotly.tools.make_subplots(rows=5, cols=2,specs=[[{'is_3d': True}, {'is_3d': True}],
														[{'is_3d': True}, {'is_3d': True}],
														[{'is_3d': True}, {'is_3d': True}],
														[{'is_3d': True}, {'is_3d': True}],
														[{'is_3d': True}, {'is_3d': True}]])

fig['layout'].update(layout)


cell_types = sorted(cell_types)
print cell_types
color1 = '#2ca02c' #green
color2 = '#ff7f0e' #orange
color = '#7f7f7f' #grey

for cell_type in cell_types:
	curr_inds = df1_names.index[df1_names['ID'].str.startswith(cell_type)]
	curr_coords = df1.values[curr_inds, 3:7].astype(float)
	orig_coords.append(curr_coords)
	names = df1.values[curr_inds, 1]
	x,y,z = zip(*curr_coords)
	if cell_type.lower() == 'd':
		trace = go.Scatter3d(x=x,y=y,z=z,mode='markers',
							marker=dict(size=5,
										line=dict(color=color,width=0.5),
										opacity=0.7,),
							name='ddn')
		all_traces.append(trace)
		fig.append_trace(trace,1,1)
	elif cell_type.lower() == 'a':
		trace = go.Scatter3d(x=x,y=y,z=z,mode='markers',
							marker=dict(size=5,
										line=dict(color=color,width=0.5),
										opacity=0.7,),
							name='ant')
		fig.append_trace(trace,1,1)

	else:
		for i in range(curr_coords.shape[0]):
			x = (curr_coords[i,0],)
			y = (curr_coords[i,1],)
			z = (curr_coords[i,2],)
			name = names[i]
			if name.split('-')[1].isalpha():
				color = color2
			else:
				color = color1
			trace = go.Scatter3d(x=x,y=y,z=z,mode='markers',
							marker=dict(size=5,
										color=color,
										line=dict(color=color,width=0.5),
										opacity=0.7,),
							name=name)
			all_traces.append(trace)

			fig.append_trace(trace, 1,1)

row=1
col=2
color1 = '#1f77b4' #blue
color2 = '#e377c2' #pink
color3 = '#9467bd' #purple
color4 = '#7f7f7f' #grey
for subfolder in os.listdir(label_folder):
	if not os.path.isdir(subfolder):
		continue

	print subfolder

	currfolder = os.path.join(label_folder, subfolder)
	reg_file = os.path.join(currfolder, 'All_cells.xls')
	if not os.path.exists(reg_file):
		reg_file = os.path.join(currfolder, 'All_cells.xlsx')
	affine_name = os.path.join(currfolder, 'Affine_Matrix.mat')

	df2 = pd.read_excel(reg_file)
	
	# load cells
	names = np.unique(df2.values[:,0])
	cellinds_vgat = df2.index[df2['neurotransmitter'].str.lower() == 'vgat']
	cellinds_vglut = df2.index[df2['neurotransmitter'].str.lower() == 'vglut']
	cellinds_both = df2.index[df2['neurotransmitter'].str.lower() == 'vgat vglut']
	cellinds_ddn = df2.index[df2['ID'].str.lower().str.contains('ddn')]
	cellinds_ant = df2.index[df2['ID'].str.lower() == 'ant']
	cellinds_other = df2.index[(pd.isnull(df2['neurotransmitter'])) & (df2['ID'].str.lower().str.startswith('pr'))]
	cells_vgat = df2.iloc[cellinds_vgat,:]
	cells_vglut = df2.iloc[cellinds_vglut,:]
	cells_both = df2.iloc[cellinds_both,:]
	cells_ddn = df2.iloc[cellinds_ddn,:]
	cells_ant = df2.iloc[cellinds_ant,:]
	cells_other = df2.iloc[cellinds_other,:]

	vgat_coords = cells_vgat.values[:, 1:4].astype(float)
	vglut_coords = cells_vglut.values[:, 1:4].astype(float)
	both_coords = cells_both.values[:, 1:4].astype(float)
	ddn_coords = cells_ddn.values[:, 1:4].astype(float)
	ant_coords = cells_ant.values[:, 1:4].astype(float)
	other_coords = cells_other.values[:, 1:4].astype(float)

	exp_coords = [vgat_coords, vglut_coords, both_coords, ddn_coords, ant_coords, other_coords]
	
	i = 0
	for coords in exp_coords:
		if coords.size == 0:
			i += 1
			continue
		x,y,z = zip(*coords)
		color = "#%06x" % random.randint(0, 0xFFFFFF)
		print color
		if i == 3:
			name = 'ddn'
			trace = go.Scatter3d(x=x,y=y,z=z,mode='markers',
								marker=dict(size=5,
											line=dict(color=color,width=0.5),
											opacity=0.7),
								name=name)
		elif i == 4:
			name = 'ant'
			trace = go.Scatter3d(x=x,y=y,z=z,mode='markers',
								marker=dict(size=5,
											line=dict(color=color,width=0.5),
											opacity=0.7),
								name=name)
		else:
			if i==0:
				name = 'vgat'
				color=color1
			elif i == 1:
				name = 'vglut'
				color=color2
			elif i == 2:
				name = 'both'
				color=color3
			elif i == 5:
				name = 'pr_none'
				color=color4
			trace = go.Scatter3d(x=x,y=y,z=z,mode='markers',
								marker=dict(size=5,
											color=color,
											line=dict(color=color,width=0.5),
											opacity=0.7),
								name=name)
		all_traces.append(trace)
		i += 1
		fig.append_trace(trace, row, col)
	if col == 2:
		col = 1
		row += 1
	else:
		col += 1

py.iplot(fig)

	

