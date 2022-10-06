import os
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.colors as pltc
import re

import seaborn as sns
from bokeh.models import Range1d, FixedTicker, Label, LinearAxis, Legend, CategoricalAxis,ColumnDataSource
from latex_label import LatexLabel
from bokeh.plotting import figure, output_file, show, save
#from bokeh.io import show
from bokeh.models import HoverTool, LabelSet
from bokeh.util.compiler import TypeScript
from bokeh import palettes

from IPython.display import display, Latex

#handle_color = '#7D007D'


# version 0.3.0
chars = {"DW_field_creep":'o',
		"DW_field_flow":'8',
		"DW_current":'h', 
		"Domain_pattern":"v",
		"Stripe_annihilation":"<",
		"Nucleation": ">",
		"BLS":'D',
		'PSWS':'d',
		'TR_MOKE' : "X",
		'MASW' : "x",
		"Spin_orbit_torque": 's'}
chars_bokeh = {"DW_field_creep":'circle', 
				"DW_field_flow":'circle_x', 
				"DW_current":'hex',  
				"Domain_pattern":"inverted_triangle",
				"Stripe_annihilation":"square_cross",
				"Nucleation": "square_dot",
				"BLS":'diamond',
				'PSWS':'diamond_cross',
				'TR_MOKE' : "star",
				'MASW' : "star_dot",
				"Spin_orbit_torque": 'triangle_pin',
				}

hatch_patterns = {"DW field creep":'/', 
				"DW field flow":'\\', 
				"DW current":'-',  
				"Domain pattern":"x",
				"Stripe annihilation":"!",
				"Nucleation": "dot",
				"Stray field": '"',
				"BLS":'@',
				'PSWS':'o',
				'TR MOKE' : "v",
				'MASW' : ">",
				"Spin orbit torque": '*',
				}

redColors = {"DW field creep":'#8B0000', #DarkRed 	
				"DW field flow":'#B22222', #FireBrick 	
				"DW current":'#FF0000',  #Red 	
				"Domain pattern":"#DC143C", #Crimson 	
				"Stripe annihilation":"#CD5C5C", #IndianRed  	
				"Nucleation": "#F08080", #LightCoral 		
				"Stray field": '#E9967A', #DarkSalmon 	
				"BLS":'#8B008B', #DarkMagenta
				'PSWS':'#800080', #Purple
				'TR MOKE' : "#EE82EE", # Violet
				'MASW' : "#FA8072",				# 	Salmon
				"Spin orbit torque": '#FFA07A', # LightSalmon
				}


greenColors = {"DW field creep":'#006400', #DarkGreen 	
				"DW field flow":'#2E8B57', #SeaGreen
				"DW current":'#FF0000',  #Green 	
				"Domain pattern":"#3CB371", #MediumSeaGreen 	
				"Stripe annihilation":"#6B8E23", #OliveDrab  	
				"Nucleation": "#32CD32", #LimeGreen 		
				"Stray field": '#00FF00', #Lime 	
				"BLS":'#556B2F', #DarkOliveGreen
				'PSWS':'#008080', #Teal
				'TR MOKE' : "#EE82EE", # Violet
				'MASW' : "#FA8072",				# 	Salmon
				"Spin orbit torque": '#FFA07A', # LightSalmon
				}

methods_with_sign = ["DW field creep", "DW field flow",
											"DW current", "Nucleation", "BLS", 'PSWS']

mm = list(greenColors.keys())
choice1 = ['rocket','mako']
choice2 = ['flare', 'crest']
choice3 = ['Blues', 'YlOrBr']
ch = choice3
# Method 1: Uniform colors (aka blues, oranges)
red_palette = sns.color_palette(ch[0], 3+len(methods_with_sign))[3:]
green_palette = sns.color_palette(ch[1], 3+len(methods_with_sign))[3:]
red_palette = red_palette[::-1]
green_palette = green_palette[::-1]

# Method 2: two colors (green+blu, orange+brown)
red_palette = red_palette[1:-1] + ['seagreen', '#96c5ab']
green_palette = green_palette[1:-1] + ['crimson', '#ed899d']
#['mediumvioletred', 'pink']





greenColors = {}
redColors = {}
for method, red, green in zip(methods_with_sign, red_palette, green_palette):
	redColors[method] = pltc.to_hex(red)
	greenColors[method] = pltc.to_hex(green)


# grayColors = {"DW field creep":'#808080', 
# 							"Domain pattern": '#696969',
# 							"Stripe annihilation": '#A9A9A9',
# 							"Nucleation":'#C0C0C0',
# 							"Spin torque": '#D3D3D3',
# 							"BLS": '#DCDCDC'}
methods_no_sign = ["DW field creep",
									"DW field flow",
									"Domain pattern",
									"Stripe annihilation",
									"Nucleation",
									"Spin orbit torque",
									"BLS"]
gray_palette = sns.light_palette('gray', n_colors=len(methods_no_sign), reverse=True)
grayColors = {}
for method, gray in zip(methods_no_sign, gray_palette):
	grayColors[method] = pltc.to_hex(gray)



fillstyle = {"DW_field_creep":['left', 'o'], 
			"DW_field_flow":['right', 'o'], 
			"DW_current":['bottom', 'o'], 
			"Domain_pattern": ["top", 's'], 
			"Stripe_annihilation": ["right", 's'],
			"Nucleation": ["left", 's'],
			#"BLS":['full','p'],
			"BLS":['full','^'],
			"PSWS":['full','v'],
			'TR_MOKE' : ['full', "X"],
			'MASW' : ['full', "x"],
			"Spin_orbit_torque": ['full','>'],
			}


colors = {1: "red", -1: "blue", 0: "black"}

marker_style = dict(linestyle='none', markerfacecoloralt='tab:grey')

#gray_methods = ['Domain_pattern', 'Stripe_annihilation', 'Nucleation', 'Spin_torque']

def get_string(s):
	#s = str(s)
	if s == int(s):
		return str(int(s))
	elif s*10 == int(s*10):
		return "%.1f" % s
	elif s*100 == int(s*100):
		return "%.2f" % s
	elif len(str(s)) > 4:
		s = str(s)[:5]
		while True:
			if s[-1] == "0":
				s = s[:-1]
			else:
				return s


TS_CODE = """
import * as p from "core/properties"
import {Label, LabelView} from "models/annotations/label"
declare const katex: any

export class LatexLabelView extends LabelView {
  model: LatexLabel

  render(): void {
    //--- Start of copied section from ``Label.render`` implementation

    // Here because AngleSpec does units tranform and label doesn't support specs
    let angle: number
    switch (this.model.angle_units) {
      case "rad": {
        angle = -this.model.angle
        break
      }
      case "deg": {
        angle = (-this.model.angle * Math.PI) / 180.0
        break
      }
      default:
        throw new Error("unreachable code")
    }

    const panel = this.layout ?? this.plot_view.layout.center_panel

    let sx = this.model.x_units == "data" ? this.coordinates.x_scale.compute(this.model.x) : panel.xview.compute(this.model.x)
    let sy = this.model.y_units == "data" ? this.coordinates.y_scale.compute(this.model.y) : panel.yview.compute(this.model.y)

    sx += this.model.x_offset
    sy -= this.model.y_offset

    //--- End of copied section from ``Label.render`` implementation
    // Must render as superpositioned div (not on canvas) so that KaTex
    // css can properly style the text
    this._css_text(this.layer.ctx, "", sx, sy, angle)

    // ``katex`` is loaded into the global window at runtime
    // katex.renderToString returns a html ``span`` element
    katex.render(this.model.text, this.el, {displayMode: true})
  }
}

export namespace LatexLabel {
  export type Attrs = p.AttrsOf<Props>

  export type Props = Label.Props
}

export interface LatexLabel extends LatexLabel.Attrs {}

export class LatexLabel extends Label {
  properties: LatexLabel.Props
  __view_type__: LatexLabelView

  constructor(attrs?: Partial<LatexLabel.Attrs>) {
    super(attrs)
  }

  static {
    this.prototype.default_view = LatexLabelView
  }
}
"""


class LatexLabel(Label):
    """A subclass of the Bokeh built-in `Label` that supports rendering
    LaTex using the KaTex typesetting library.

    Only the render method of LabelView is overloaded to perform the
    text -> latex (via katex) conversion. Note: ``render_mode="canvas``
    isn't supported and certain DOM manipulation happens in the Label
    superclass implementation that requires explicitly setting
    `render_mode='css'`).
    """
    __javascript__ = ["https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.6.0/katex.min.js"]
    __css__ = ["https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.6.0/katex.min.css"]
    __implementation__ = TypeScript(TS_CODE)



class DMI:
	def __init__(self, methods, filename="DMI.xlsx", papers="dmi_paper.xlsx", is_tofix=True):
		self.figs = []
		self.xl = pd.ExcelFile(filename, engine='openpyxl')
		self.sheetnames = [sname for sname in methods if sname in self.xl.sheet_names]
		self.is_tofix = is_tofix
		self.fmats = []
		self.k_factor = {"Ds": "p", "D": "m"}
		self.n_experiments = {}
		# Create the full data dataframe
		all_data = None
		for sname in self.sheetnames:
			if all_data is None:
				all_data = self.xl.parse(sname, dtype={"composition": str})
				all_data['Method'] = len(all_data) * [sname]
			else:
				_data = self.xl.parse(sname, dtype={"composition": str})
				if _data.empty:
					continue
				_data['Method'] = len(_data) * [sname]
				all_data = pd.concat((all_data,_data), sort=False)
		all_data = all_data.fillna("")
		self.all_data_df = all_data.loc[:,'fm':'Method']
		self.papers = pd.read_excel(papers, index_col=0)
		self.set_authors()

	def set_authors(self):
		self.papers.Author

	def print_most_popular_stacks(self, fm=None,nfirst=15):
		data = self.all_data_df[self.all_data_df.Note!='To fix']
		if fm == None:
			q = data.groupby(['top','fm','bottom']).size().sort_values(ascending=False)[:nfirst]
		else:
			data = data[data.fm==fm]
			#print(fm)
			q = data.groupby(['top','bottom']).size().sort_values(ascending=False)[:nfirst]
		q = q.to_frame()
		q.columns = ['n']
		q = q.T
		#display(q)
		return q
		
	def plotStack(self, stack, select_D="Ds", 
					size_factor=30, alpha=0.5, 
					figsizeX=8, marker_type='shapes',
					marker_colors=colors, handle_color='gray',
					set_axis=None, ndots=20,
					visualization_library="mpl", is_XY_to_plot=False, is_anular_to_plot=True):
		len_bottom, len_fm, len_top = [len(st) for st in stack]
		try:
			assert len_bottom==1 or len_top==1
		except AssertionError:
			print("At least top/bottom must be a single layer")
			sys.exit()
		if len_top + len_bottom == 2:
			fig_out = self._plotTopBottom(stack, select_D, 
					size_factor, alpha, figsizeX,
					marker_type, marker_colors, handle_color, set_axis,
					ndots, visualization_library)
			print("Plot Top Botttom")
			if visualization_library == 'bokeh' and is_plot:
				show(fig_out, notebook_handle=True)
			return fig_out
		else:
			fig_out, fig_anular = self._elementsVsThickness(stack, select_D, 
					size_factor, alpha, 
					figsizeX, marker_type, marker_colors, handle_color, set_axis,
					ndots, visualization_library)
			#print("Plot Element vs Thickness")
			if visualization_library == 'bokeh' and is_XY_to_plot:
				show(fig_out, notebook_handle=True)
			if fig_anular is not None:
				show(fig_anular, notebook_handle=True)
			return fig_out, fig_anular
		

	def _get_elements(self, stack):
		bottom, fmats, top = stack
		bottom_elements, bottom_thickness = self._get_thickness(bottom)
		top_elements, top_thickness = self._get_thickness(top)
		_fmats, _fmats_composition = [], []
		for fmat in fmats:
			fmat_composition = "".join(re.findall("[0-9]", fmat))
			if len(fmat_composition):
				fmat = "".join(re.findall("[a-zA-Z]", fmat))
			_fmats.append(fmat)
			_fmats_composition.append(fmat_composition)
		out = [bottom_elements, bottom_thickness, _fmats, _fmats_composition, 
			top_elements, top_thickness]
		return out
		
	def _elementsVsThickness(self, stack, select_D="Ds", 
							size_factor=100, alpha=0.2, 
							figsizeX=8, marker_type='fillings',
							marker_colors=colors, handle_color='gray',
							set_axis=None, ndots=20,
							visualization_library='mpl',
							show_annular_plot=True):
		"""
		pass the stack to be analysed
		([bottom elements], [top elements])
		Example: 
			(['Hf', 'Ta', 'TaN','W', 'IrMn'], ['CoFeB'], ['MgO'])

		It is possible to select a particular thickness in nm, like
			(['Hf', 'Ta', 'TaN','W', 'IrMn'], ['MgO(2)'])
		or give a range, or select some values
			(['Hf(2,3)', 'Ta(1-3)', 'TaN','W', 'IrMn'], ['MgO(2)'])

		marker_type: strings
			shapes|fillings

		"""
		if isinstance(figsizeX, tuple):
			figsize = figsizeX
		else:
			figsize = (figsizeX*2.2, figsizeX*1.2)
		bottom, fmats, top = stack
		out = self._get_elements(stack)
		[bottom_elements, bottom_thickness, fmats, fmats_composition, 
			top_elements, top_thickness] = out
		if len(top_elements) == 1:
			fixed_layer_element = top_elements[0]
			fixed_layer_thickness = top_thickness[0]
			fixed_layer = "top"
			fixed_tlayer = "ttop"
			elements = bottom_elements[::-1]
			telements = bottom_thickness[::-1]
			layer = "bottom"
			tlayer = "tbottom"
			y_labels = bottom[::-1]
			self.multiple_layer = 'bottom'
		elif len(bottom_elements) == 1:
			fixed_layer_element = bottom_elements[0]
			fixed_layer_thickness = bottom_thickness[0]
			fixed_layer = "bottom"
			fixed_tlayer = "tbottom"
			elements = top_elements[::-1]
			telements = top_thickness[::-1]
			layer = "top"
			tlayer = "ttop"
			y_labels = top[::-1]
			self.multiple_layer = 'top'
		else:
			raise("Too many elements")
		_elements = [r"${}$".format(elem.replace("x", "_x")) for elem in elements]
		self.data = None
		for fmat, fmat_composition in zip(fmats, fmats_composition):
			if visualization_library == 'mpl':
				fig, ax = plt.subplots(1,1,figsize=figsize)
				ax.set_position([0.12,0.12,.65,.75])
				
			n_data = 0
			##################################################
			self.dic_n_elements = {}
			sheetnames_with_data = self.sheetnames.copy()
			markers = []
			methods = []
			Ds_strings = []
			for sname in self.sheetnames:
				self.n_elements = []
				data = self.xl.parse(sname)
				data = data[data.fm == fmat]
				data = data[data[fixed_layer]==fixed_layer_element]
				#print(sname, data.shape)
				if len(fixed_layer_thickness):
					if "-" in fixed_layer_thickness:
						v1, v2 = fixed_layer_thickness.split("-")
						v1, v2 = float(v1), float(v2)
						is_to_select = data[fixed_tlayer].between(v1, v2, inclusive=True)
					else:
						is_to_select = data[fixed_tlayer] == float(fixed_layer_thickness)
					data = data[is_to_select]
				if fmat_composition:
					data = data[data.composition == int(fmat_composition)]
				#print("********** %s ***********" % sname)
				#print(data)
				# Skip an empty dataframe
				if data.size == 0:
					sheetnames_with_data.remove(sname)
					continue
				for y, _e in enumerate(zip(elements, telements)):
					element, t_element = _e
					n_data = 0
					is_to_select = data[layer] == element
					sub_data = data[is_to_select]
					if t_element != "":
						if "-" in t_element:
							v1, v2 = t_element.split("-")
							v1, v2 = float(v1), float(v2)
							is_to_select = sub_data[tlayer].between(v1, v2, inclusive=True)
						else:
							is_to_select = data[tlayer] == telement
						sub_data = sub_data[is_to_select]
					#print(sub_data)
					if sub_data.size == 0:
						self.n_elements.append(0)
						continue
					#else:
						#print("*** %s ***" % element)
						#print(sub_data)
					if self.is_tofix:
						_s = [str(d) for d in sub_data.Note]
						is_to_select = [_ss != 'To fix' for _ss in _s]
						sub_data = sub_data[is_to_select]
					if self.data is None:
						self.data = sub_data.copy()
					else:
						self.data = pd.concat((self.data,sub_data))
					D_data = sub_data[select_D]
					notes = sub_data.Note
					#print(sname, _e)
					for i in sub_data.index:
						Ds = D_data.loc[i]
						x = sub_data[tlayer].loc[i]
						fill_color = marker_colors[np.sign(Ds)]
						#print(x, y, Ds)
						size = np.abs(Ds)**0.5 * size_factor
						m = chars[sname]
						markers.append(chars_bokeh[sname])
						methods.append(sname.replace("_"," "))
						if Ds:
							if marker_type == 'shapes':
								is_sign_given = sub_data.is_sign_given.loc[i]
								#_filter1 = sname in ['Domain_pattern', 'Stripe_annihilation', 'Spin_torque']
								# _filter2 = (sname == 'Nucleation') and (notes.loc[i] == 'Bubble')
								if is_sign_given:
									_c, _alpha, edgecolors = fill_color, alpha, 'face'
									Ds_string = f"{Ds:.2f} pJ/m"
								else:
									_c, _alpha, edgecolors = 'lightgray', .5, 'black'
									Ds_string = f"|{Ds:.2f}| pJ/m"
								Ds_strings.append(Ds_string)
								if visualization_library == 'mpl':
									ax.scatter(x, y+1, c=_c, s=size**2, alpha=_alpha, marker=m, 
												edgecolors=edgecolors, linewidths=2)
							elif marker_type == 'fillings':
								color = "tab:%s" % fill_color
								fstyle, marker = fillstyle[sname]
								ax.plot(x, y+1, markersize=size, alpha=alpha, fillstyle=fstyle, marker=marker,
									color=color, mew=2, **marker_style)
						else:
							if marker_type == 'shapes':
								if visualization_library == 'mpl':
									ax.scatter(x, y+1,c='black',s=10,alpha=alpha,marker=m)
								else:
									Ds_string = f"0 pJ/m"
									Ds_strings.append(Ds_string)
							elif marker_type == 'fillings':
								ax.plot(x, y+1, markersize=size_factor, alpha=1, fillstyle='full', color='black', mew=2)
						
						n_data += 1
					self.n_elements.append(n_data)
				self.dic_n_elements[sname] = self.n_elements
				#print(dic_n_elements)
				#print("Loop done")
			#print("Full Loop done")
			
			df = pd.DataFrame(self.dic_n_elements, index=elements)
			if df.empty:
				print("No data available")
				continue    
			df.loc["Total"] = df.sum(axis=0)
			stack_key = self._stack_as_string(stack)
			self.n_experiments[stack_key] = df
			self.fmats.append(fmat)
			title = self._get_title_stack(fmat, fmat_composition, fixed_layer, fixed_layer_element, fixed_layer_thickness, select_D)
			xlabel = "%s layer thickness (nm)" % layer
			ylabel = "%s layer" % layer

			if visualization_library == 'mpl':
				if set_axis is not None:
					ax.set_xlim(set_axis[:2])
					ax.set_ylim(set_axis[2:])
				else:
					ax.set_ylim(0.2, len(elements)+0.8)
					ax.set_yticks(range(1,len(elements)+1))
				ax.set_yticklabels(_elements, size=20)
				ax.set_xlabel(xlabel, size=20)
				ax.set_ylabel(ylabel, size=20)
				# Make the legends
				self._print_legends(sheetnames_with_data, marker_type, marker_colors, select_D, 
									size_factor, alpha, handle_color, zero_point=False)
				ax.set_title(title, size=20)
				plt.show()
			elif visualization_library == 'bokeh':
				self.data = self.data.fillna("")
				
				items = ['Author', 'Title', 'Journal', 'Volume', 'Pages', 'Year']
				for item in items:
					self.data.loc[:, item] = [self.papers.loc[ref][item] for ref in self.data.Ref]
					# self.data[item] = [self.papers.loc[ref][item] for ref in self.data.Ref]
				self.data.loc[:,'Author'] = self.get_authors_string(self.data['Author'])
				self.data.loc[:,'size'] = np.abs(self.data.Ds)**0.5 * size_factor
				self.data.loc[:,'alpha'] = [alpha if sign else 0.5 for sign in self.data.is_sign_given]
				self.data.loc[:,'marker'] = markers
				self.data.loc[:,'method'] = methods
				_line_colors, _fill_colors = [], []
				for sign, Ds in zip(self.data.is_sign_given, self.data.Ds):
					if sign:
						_c = marker_colors[np.sign(Ds)]
						_line_colors.append(_c)
						_fill_colors.append(_c)
					else:
						_line_colors.append('black')
						_fill_colors.append('lightgray')
				self.data.loc[:,'line_color'] = _line_colors
				self.data.loc[:,'fill_color'] = _fill_colors
				self.data.loc[:,'Ds_string'] = Ds_strings
				stack_string = self.get_stack_string(self.data)
				self.data.loc[:,'stack_string'] = stack_string
				tooltips = [('Stack', '@stack_string'),
							('D_s', '@Ds_string'), 
							('Method', '@method'),
							('Authors', '@Author'),
							('Tile', '@Title'),
							('Ref', '@Journal, @Volume, @Pages (@Year)'),
							('', 20*'-')]
				fig = figure(plot_width=figsize[0], plot_height=figsize[1],
							y_range=y_labels, tooltips=tooltips, toolbar_location='above')
				fig.add_layout(CategoricalAxis(), 'right')
				fig.add_layout(Legend(), 'right')
				fig.scatter(f"t{layer}", layer, size='size', alpha='alpha', source=self.data, 
								line_color='line_color', fill_color='fill_color', marker='marker',
								legend_group='method')
				fig.xaxis.axis_label = xlabel
				fig.xaxis.axis_label_text_font_size = "25px"
				# yaxis
				fig.yaxis.axis_label = ylabel
				fig.yaxis.axis_label_text_font_size = "20px"
				fig.yaxis.axis_label_text_font_style = "italic"
				fig.yaxis.major_label_text_font_size = "25px"
				fig.yaxis.major_label_text_font_style = "italic"
				fig.add_layout(LinearAxis(), 'above')
				#fig.extra_y_ranges = {"right": fig.y_range}
				
				# if set_axis is not None:
				#     left, right, bottom, top = set_axis
				#     fig.x_range = Range1d(left, right)
				# else:
				#     pass
					#fig.y_range = Range1d(1, len(elements)+1)
					#fig.xaxis.ticker = FixedTicker(ticks=list(range(1,len(elements)+1)))
				#size=20
				#ax.set_xlabel("%s layer thickness (nm)" % layer, size=20)
				#ax.set_ylabel("%s layer" % layer, size=20)
			# Make the legends
				#self._print_legends(sheetnames_with_data, marker_type, marker_colors, select_D, 
				#    size_factor, alpha, handle_color, zero_point=False)
				fig.legend.title = 'Methods'
				fig.legend.label_text_color = 'black'
				title = title.replace("$","")
				fig.title.text = title.replace("_","")
				fig.title.text_font_size = "25px"
				t0, t1 = title.split("-")
				title_stack = t1
				t1 = t1.replace("/", "_").strip()
				t1 = f"{t1}.html"
				#output_file(t1, title=t1)
				if show_annular_plot:
					fig_annular = self.plot_annular(self.data, elements, telements,
																					title=title_stack, xlabel=xlabel)
					return fig, fig_annular
		return fig, None

	def plot_annular(self, df_data, elements, telements, 
									max_thickness=6, step_tickness=0.5, xlabel=None, title=None):
		self.df_annular = df_data.copy()
		self.df_annular = self.df_annular.sort_values('Ds', ascending=False, key=np.abs)
		# Set the layer with different elements
		tlayer = "t" + self.multiple_layer
		self.df_annular = self.df_annular[self.df_annular[tlayer] < max_thickness]

		type_elements = {
			'metals': ['W', 'Pt', 'Ir', 'Au', 'Ta', 'Al', 'Gd', 'IrMn', 'Hf', 'TaN', 'Cu'],
			'oxides': ['AlOx', 'GdO', 'MgO']
			}

		elements_color = dict([
			#("metals", "#e69584"),
			#("metals", "#f0f0f0"),
			("metals", "#f5fffa"), #Mintcream
			#("metals", "aliceblue"),
			#("oxides", "#aeaeb8"),
			#("oxides", "#fee8ca"),
			("oxides", "cornsilk"),
			])

		metals_selected = False
		oxides_selected = False
		colors = []
		for element in elements:	
			if element in type_elements['metals']:
				colors.append(elements_color['metals'])
				metals_selected = True
			elif element in type_elements['oxides']:
				colors.append(elements_color['oxides'])
				oxides_selected = True
			else:
				print('Element %s non available' % element)
				colors.append('#000000') #

		#greens = ['#002510'] + list(palettes.Greens9)[:-1]
		#oranges = ['#5c1a09'] + list(palettes.Oranges9)[:-1]
		#greens = list(palettes.Greens9)
		#oranges = list(palettes.Oranges9)


		methods_colors_red = [redColors[m] for m in methods_with_sign]
		methods_colors_green = [greenColors[m] for m in methods_with_sign]
		methods_colors_gray = [grayColors[m] for m in methods_no_sign]

		
		clrs = []
		for Ds, sign, method in zip(self.df_annular.Ds, self.df_annular.is_sign_given, self.df_annular.method):
			if sign:
				#i = methods.index(method)
				if Ds > 0:
					#c = greens[i]
					c = greenColors[method]
				elif Ds < 0:
					#c = oranges[i]
					c = redColors[method]
			else:
					c = grayColors[method]
			clrs.append(c)
		self.df_annular.loc[:,'fill_color'] = clrs


		width = 800
		height = 800
		inner_radius = 70
		outer_radius = 270
		Y0 = 30
		

		minr, maxr = 0, max(abs(self.df_annular.Ds))
		a = (outer_radius - inner_radius) / (maxr - minr)
		
		self.df_annular.loc[:,'rads'] = inner_radius + a * np.abs(self.df_annular.Ds)
		rads = inner_radius + a * np.abs(self.df_annular.Ds)

		n_sectors = len(elements)
		print(f"{n_sectors} sectors to plot: {elements}")
		big_angle = 2.0 * np.pi / (n_sectors + 1)
		n_small_angle = max_thickness / step_tickness
		small_angle = big_angle / n_small_angle
		
		tooltips = [('Stack', '@stack_string'),
							('D_s', '@Ds_string (@method)'), 
							('Paper', '@Author, @Title, @Journal @Volume, @Pages (@Year)'),
							('', 20*'-')]

		#[output_file(filename="custom_filename.html", title="Static HTML file")

		p = figure(width=width, height=height, title="",
			x_axis_type=None, y_axis_type=None,
			x_range=(-350, 350), y_range=(-350, 350),
			min_border=0, outline_line_color="black",
			#background_fill_color="#f0e1d2")
			tooltips=tooltips,
			tools=['pan', 'box_zoom', 'wheel_zoom', 'save',
                                 'reset', 'hover'],
			background_fill_color="#ffffff")

		hover_tool = p.select(type=HoverTool)
		hover_tool.names = ["DsValues"]

		p.axis.visible = True
		p.xgrid.grid_line_color = None
		p.ygrid.grid_line_color = None

		# annular wedges
		n_sector = np.array([elements.index(e) for e in self.df_annular[self.multiple_layer]])
		_angles = np.pi/2 - big_angle/2 - n_sector * big_angle
		# Calculate the centers of the wedge in respect the sector
		c0 = big_angle / max_thickness * self.df_annular[tlayer]
		self.df_annular.loc[:,'angle0'] =  _angles - c0 + small_angle/2
		self.df_annular.loc[:,'angle1'] =  self.df_annular['angle0'] - small_angle

		#self.df_annular.loc[:,'hatch_patterns'] = [hatch_patterns[method] for method in self.df_annular['method']]

		#angle0 =  -big_angle + _angles + (self.df_annular[tlayer]/step_tickness) * small_angle
		#angle1 =  angle0 + small_angle


		#self.df_annular.loc[:,'angle1'] = self.df_annular['fill_color']

		angles = np.pi/2 - big_angle/2 - np.arange(n_sectors+1) * big_angle
		p.annular_wedge(0, Y0, inner_radius, outer_radius*1.02, -big_angle+angles, angles, color=colors,
										inner_radius_units='data', outer_radius_units='data')
		# radial axes
		p.annular_wedge(0, Y0, inner_radius-8, outer_radius+10, 
									-big_angle+angles, -big_angle+angles, 
									color="black", inner_radius_units='data', outer_radius_units='data')
		ticks = big_angle / max_thickness * np.arange(1,max_thickness*(n_sectors+1))
		ticks_angles = np.pi/2 - big_angle/2 - ticks
		p.annular_wedge(0, Y0, inner_radius-5, inner_radius, ticks_angles, ticks_angles, 
										color="black",inner_radius_units='data', outer_radius_units='data')

		R0 = inner_radius *.8
		ticks_alpha =  np.pi/2 + big_angle/2 - big_angle/max_thickness * np.arange(0,max_thickness+2,2)
		ns = [str(n) for n in np.arange(0,max_thickness+2,2)]
		# add other x_ticks if liked
		if False:
			_ticks_alpha = np.copy(ticks_alpha)
			sector_numbers = [4,7]
			ns = len(sector_numbers) * ns
			for i in sector_numbers:
				_alpha = _ticks_alpha + i * big_angle
				ticks_alpha = np.concatenate((ticks_alpha, _alpha))
		source_label = ColumnDataSource(data=dict(x=R0*np.cos(ticks_alpha), 
																							y=R0*np.sin(ticks_alpha),
																							tick_angles = (-np.pi/2 + ticks_alpha),
																							ns = ns)
																		)
		lbtk = LabelSet(x='x',y='y',y_offset=Y0, 
										text_font_size = "8px", text='ns', angle='tick_angles', source=source_label,
										text_align='center')
		p.add_layout(lbtk)
		#angle = -big_angle + angles[1]
		#lbtk1 = Label(x=R0*np.cos(angle), y=R0*np.sin(angle),y_offset=Y0, 
		#								text_font_size = "10px", text=str(max_thickness), angle=angle-np.pi/2)
		#p.add_layout(lbtk1)


		# small wedges
		p.annular_wedge(0, Y0, inner_radius=inner_radius, outer_radius='rads',
		  				start_angle='angle0', end_angle='angle1',
		  				fill_color='fill_color', 
		  				line_color='black',
		  				hatch_color=None,
		  				#hatch_pattern='hatch_patterns',
		  				#hatch_scale=.5,
		  				name = "DsValues",
		  				direction='clock',
		  				source=self.df_annular,
		  				inner_radius_units='data', outer_radius_units='data')


		# circular axes and lables

		n_radii = int(maxr / 0.5) + 1
		labels = np.arange(n_radii) * 0.5
		radii = inner_radius + a * labels
		

		p.circle(0, Y0, radius=radii, fill_color=None, line_color="grey", radius_units='data')
		p.arc(0, Y0,radius=outer_radius*1.02, start_angle = np.pi/2 + big_angle/2, end_angle = np.pi/2 - big_angle/2, 
					radius_units='data', color='black')
		for i, r in enumerate(labels):
		 		p.add_layout(Label(x=0, y=radii[i]+5*np.exp(-i/2), y_offset=Y0, text=str(r), x_units='data', y_units='data',
		 			  text_font_size="10px", text_align="center", text_baseline="bottom"))


	
		# labels of elements
		xr = 1.03 * outer_radius*np.cos(np.array(-big_angle/2 + angles[:-1]))
		yr = 1.03 * outer_radius*np.sin(np.array(-big_angle/2 + angles)[:-1]) + Y0
		#p.circle(xr, yr, radius=2, fill_color='gray', line_color="grey")
		label_angle=np.array(-big_angle/2+angles[:-1])
		selection = label_angle < -np.pi/2
		i = np.argmax(selection)
		label_angle[selection] += np.pi # easier to read labels on the left side
		p.text(xr[:i], yr[:i], elements[:i], angle=label_angle[:i],
		 	   text_font_size="20px", text_align="left", text_baseline="middle")
		p.text(xr[i:], yr[i:], elements[i:], angle=label_angle[i:],
		 	   text_font_size="20px", text_align="right", text_baseline="middle")



		# Color patterns
		h = outer_radius + 15
		x0 = -h
		y0 = -h-5
		l0 = 15
		dy = 30

		xs = x0 + l0 * np.arange(len(methods_with_sign))
		ys = np.array(len(methods_with_sign) * [y0+l0])
		p.rect(xs, ys, width=l0, height=l0, line_color='black',fill_color=methods_colors_red)
		p.text(xs[-1] + 15, ys[-1], text=['Negative DMI'], text_font_size="14px", 
						text_align="left", text_baseline="middle")

		ys = ys -l0
		p.rect(xs, ys, width=l0, height=l0, line_color='black',fill_color=methods_colors_green)
		p.text(xs[-1] + 15, ys[-1], text=['Positive DMI'], text_font_size="14px", 
						text_align="left", text_baseline="middle")

		p.text(xs, ys[-1] + 2*l0, text=methods_with_sign, angle=np.pi/2, text_font_size="14px", 
						text_align="left", text_baseline="middle")				

		n_methods_no_sign = len(methods_no_sign)
		xs = x0 + l0 * np.arange(len(methods_no_sign)) + 500
		ys = np.array(n_methods_no_sign * [y0])
		
		p.text(xs[0] - l0, ys[-1], text=['|DMI|'], text_font_size="14px", 
						text_align="right", text_baseline="middle")
		
		p.rect(xs, ys, width=l0, height=l0, line_color='black',fill_color=methods_colors_gray[::-1])
		p.text(xs, ys[-1] + l0, text=methods_no_sign[::-1], angle=np.pi/2, text_font_size="14px", 
						text_align="left", text_baseline="middle")		




		# # OK, these hand drawn legends are pretty clunky, will be improved in future release
		x0 += 5
		y0 = h - 10
		if metals_selected:
			p.rect(x0, y0, width=3*l0, height=l0,
  			     color=elements_color['metals'], line_color='black')
			lbm = Label(x=x0+2*l0, y=y0, text='Metals', text_font_size="16px", 
									text_align="left", text_baseline="middle")
			p.add_layout(lbm)
		
# Add the stack at the top of the plot
		if title:
				lb_title = Label(x=0, y=y0+55, text=title, text_font_size="30px",
									text_align="center", text_baseline="middle")
				# lb_title = LatexLabel(x=0, y=y0+55, text=title, text_font_size="30px",
				# 	    			x_units="data", y_units="data",
				# 					render_mode = 'css', background_fill_alpha=0,
				# 					text_align="center", text_baseline="middle")

				p.add_layout(lb_title)
		


		if oxides_selected:
			x0 += 530
			p.rect(x0, y0, width=3*l0, height=l0,
  			     color=elements_color['oxides'], line_color='black')
			lbo = Label(x=x0+2*l0, y=y0, text='Oxides', text_font_size="16px",
									text_align="left", text_baseline="middle")
			p.add_layout(lbo)

		if xlabel:
			self.write_curved_text(xlabel, p, R=inner_radius-25, y_offset=Y0)		

		#show(p)
		#save(p)
		return p

	def write_curved_text(self, text, fig, white_spaces=0, R=60, y_offset=None, is_curved=False):
		"""
		is_curved: letter, word, None
		"""
		
		text = white_spaces * " " + text + white_spaces * " "
		ltext = len(text)	
		if is_curved=='letter':
			angles = np.pi * (1 - (np.arange(ltext) + 1.5)/ltext) 
			for i, angle in enumerate(angles):
				if text[i]:
					lb = Label(x=R*np.cos(angle), y=R*np.sin(angle),y_offset=y_offset, 
										text_font_size = "10px", text_font = 'helvetica',
										text=text[i], angle=angle-np.pi/2)
					fig.add_layout(lb)
		elif is_curved == 'word':
			split_text = text.strip().split(" ")
			half_lens_text = [len(l)/2 for l in split_text]
			for i in range(len(split_text)):
				w = 2 * sum(half_lens_text[:i]) + half_lens_text[i] + i
				angle = np.pi * ( 1 - w / len(text))
				lb = Label(x=R*np.cos(angle), y=R*np.sin(angle),y_offset=y_offset, 
										text_font_size = "14px", text_font = 'times', text_font_style='italics',
										text=split_text[i], angle=angle-np.pi/2,text_align="center")
				fig.add_layout(lb)
		else:
			text = text.strip().split(" ")
			t1 = " ".join(text[:2])
			t2 = " ".join(text[2:])
			x1, y1 = 0, 10
			for tx in [t1,t2]:
				lb = Label(x=x1, y=y1, y_offset=y_offset,	 
										text_font_size = "20px", text_font = 'times', text_font_style='italic',
										text=tx, text_align="center")
				y1 += -20
				fig.add_layout(lb)


	def get_stack_string(self, data):
		stack_string = []
		for q in zip(data.bottom1,data.tbottom1,
					data.bottom, data.tbottom,
					data.fm, data.tfm,
					data.top, data.ttop,
					data.top1, data.ttop1, data.Repetitions):
			b1,tb1,b,tb,_fm,_tfm,t,tt,t1,tt1, rep = q
			rep = str(rep)
			if len(rep) > 2: 
				_s = rep
			else:
				_s = ""
				if b:
					_s += "%s(%s)/" % (b, get_string(tb))
				_s += "%s(%s)" % (_fm, get_string(_tfm))
				if t:
					_s += "/%s(%s)" % (t, get_string(tt))
				if rep !="":
					_s = "[%s]x%s" % (_s, rep)
				if b1:
					_s = "%s(%s)/" % (b1, get_string(tb1)) + _s
				if t1:
					_s += "/%s(%s)" % (t1, get_string(tt1))
			stack_string.append(_s)
		return stack_string

	def get_authors_string(self, authors):
		_auth = []
		for auth in authors:
			_authors = auth.split(";")
			if len(_authors) > 3:
				surname, name = _authors[0].split(",")
				name = name.strip().capitalize()
				_s = "%s, %s. et al." % (surname, name[0]) 
			else:
				_s = auth
			_auth.append(_s)
		return _auth

	def _get_thickness(self, elements):
		elements_name, elements_thickness = [], []
		for element in elements:
			thickness = re.findall("[(\-,0-9)]", element)
			thickness = "".join(thickness[1:-1])
			elements_thickness.append(thickness)
			q = element.replace("(%s)" % thickness, "")
			elements_name.append(q)
		return elements_name, elements_thickness

	def _print_legends(self, sheetnames_with_data, marker_type, marker_colors, select_D, 
						size_factor, alpha, handle_color, zero_point):
		handles1, handles2 = self._get_handles(sheetnames_with_data, marker_type, marker_colors, select_D, 
								size_factor, alpha, handle_color, zero_point)
		h_leg = self._h_leg(sheetnames_with_data)
		leg2 = plt.legend(handles=handles2, bbox_to_anchor=(1.02, 1-h_leg), loc=2, borderaxespad=0.,
						ncol=1, title=r"$\bf{DMI\ values}$", labelspacing=0.8)
		leg = plt.legend(handles=handles1, bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0., 
						ncol=1, title = r"$\bf{Methods}$")
		plt.gca().add_artist(leg2)    
		#plt.draw()
		# q = leg.get_window_extent()
		# self.p = ax, leg
		# print(q.height)
		# h_leg = q.height/h
		# plt.draw()
		# self.leg = leg
		# qq = leg.get_frame()
		# h_leg = qq.get_bbox()
		# print(h_leg)


	def _get_handles(self, snames, marker_type, marker_colors, select_D, 
					size_factor, alpha, handle_color,zero_point=False):
		handles1, handles2 = [], []
		if marker_type == 'shapes':
			for sname in snames:
				h = mlines.Line2D([], [], color=handle_color, marker=chars[sname],
						  markersize=10, label=sname.replace("_", " "), ls="None", alpha=1)
				handles1.append(h)
		elif marker_type == 'fillings':
			for sname in snames:
				fstyle, marker = fillstyle[sname]
				h = mlines.Line2D([], [], color=handle_color, markersize=10, marker=marker,
					fillstyle=fstyle, label=sname.replace("_", " "), ls="None", alpha=alpha, **marker_style)
				handles1.append(h)
		for s in ["0.2", "0.5", "1"]:
			dlabel = r"$%s$ = %s %sJ/m" % (select_D.replace("s", "_s"), s, self.k_factor[select_D])
			size = float(s)**0.5*size_factor
			size_max = plt.Line2D((0,1),(0,0), color=marker_colors[1], marker='o', alpha=alpha,
						markersize=size, linestyle='', label=dlabel)
			handles2.append(size_max)
		for c, s in zip(marker_colors.values(), ['>','<', "="]):
			dlabel = r"$%s$ %s 0" % (select_D.replace("s", "_s"), s)
			if (s == "=") and (not zero_point):
				dlabel = "No sign available"
				alpha = 0.2

			size_max = plt.Line2D((0,1),(0,0), color=c, marker='o', 
						markersize=0.5*size_factor, linestyle='', label=dlabel, alpha=alpha)
			handles2.append(size_max)
		return handles1, handles2


	def print_experiments(self):
		for key in self.n_experiments:
			#print(key)
			print("%s, %s, %s" % tuple([k.split("_") for k in key.split("__")]))
			print(self.n_experiments[key].to_string())
			print(80*"*")



	
	def print_stack_tables(self, stack, useJupyter=False):
		layers = dict()
		out = self._get_elements(stack)
		[bottom_elements, bottom_thickness, fmats, fmats_composition, 
			top_elements, top_thickness] = out
		layers['bottom'] = bottom_elements, bottom_thickness
		layers['top'] = top_elements,top_thickness
		if len(bottom_elements) > len(top_elements):
			elements = bottom_elements
			element_name = "bottom"
			fixed_layer = top_elements[0]
			fixed_layer_name = "top"
		else:
			elements = top_elements
			element_name = "top"
			fixed_layer = bottom_elements[0]
			fixed_layer_name = "bottom"

		k = 0
		for fmat, fmat_composition in zip(fmats, fmats_composition):
			data = self.all_data_df.groupby(['fm']).get_group(fmat)
			if len(fmat_composition):
				data = data.groupby(['composition']).get_group(fmat_composition)
				fmat_string = re.findall('[A-Z][^A-Z]*', fmat)
				fmatc_string = [fmat_composition[i:i+2] for i in range(0,len(fmat_composition),2)]
				fmat_string = "".join([a+b for a,b in zip(fmat_string, fmatc_string)])
				fmat_string = stack[1][k]
			else:
				fmat_string = fmat
			data = data.groupby([fixed_layer_name]).get_group(fixed_layer)
			for element in elements:
				_data = data.groupby([element_name]).get_group(element)
				if fixed_layer_name == 'bottom':
					s = "{0}/{1}/{2}".format(fixed_layer, fmat_string, element)
				else:
					s = "{0}/{1}/{2}".format(element, fmat_string, fixed_layer)
				out = "{0} {1} {0}".format(20*"*", s)
				_data = _data.fillna("")
				if useJupyter:
					out = Latex("$$%s$$" % s)
					display(out)
					display(_data)
				else:
					print(out)
					print(_data)
			k += 1


	def _plotTopBottom(self, stack, select_D="Ds", 
					size_factor=100, alpha=0.2, 
					figsizeX=6, marker_type='fillings',
					marker_colors=colors, handle_color='lightgray',
					set_axis=None, ndots=20,
					visualization_library='mpl',use_latex=False):
		"""
		plot top layer vs bottom layer
		"""
		if isinstance(figsizeX, tuple):
			figsize = figsizeX
		else:
			scale = 1.25
			figsize = (figsizeX*scale, figsizeX)
		xaxis = 0.6
		n_data = 0
		font_size = 22
		bottom, fmats, top = stack
		index = []
		if isinstance(bottom, list):
			bottom, = bottom
		if isinstance(top, list):
			top, = top
		self.data = None
		for fmat in fmats:
			fmat_composition = "".join(re.findall("[0-9]", fmat))
			if len(fmat_composition):
				fmat = "".join(re.findall("[a-zA-Z]", fmat))
			#print(fmat, fmat_composition)
			if visualization_library == 'mpl':
				fig, ax = plt.subplots(1,1,figsize=figsize)
				ax.set_position([0.125,0.125,xaxis,xaxis*1.25])
			_maxs = []
			dic_n_elements = {}
			sheetnames_with_data = self.sheetnames.copy()
			markers = []
			methods = []
			Ds_strings = []
			for sname in self.sheetnames:
				n_elements = []
				data = self.xl.parse(sname)
				data = data[data.fm == fmat]
				data = data.sort_values(by=['tfm'])
				data = data[data.bottom == bottom]
				data = data[data.top == top]
				if fmat_composition:
					data = data[data.composition == int(fmat_composition)]
				if self.is_tofix:
					_s = [str(d) for d in data.Note]
					is_to_select = [_ss != 'To fix' for _ss in _s]
					data = data[is_to_select]
				if data.size == 0:
					#print("No data for %s in %s/%s/%s" % (sname, bottom, fmat, top))
					sheetnames_with_data.remove(sname)
					continue
				D_data = data[select_D]
				notes = data.Note
				_maxs.append(np.max([data.ttop.max(), data.tbottom.max()]))
				if self.data is None:
					self.data = data.copy()
				else:
					self.data = self.data.append(data)
				for i, thickness in zip(data.index, data.tfm):
					Ds = D_data.loc[i]
					x, y = data.tbottom.loc[i], data.ttop.loc[i]
					size = np.abs(Ds)**0.5 * size_factor
					fill_color = marker_colors[np.sign(Ds)]
					m = chars[sname]
					markers.append(chars_bokeh[sname])
					methods.append(sname.replace("_"," "))
					if Ds:
						if marker_type == 'shapes':
							is_sign_given = data.is_sign_given.loc[i]
							#_filter1 = sname in ['Domain_pattern', 'Stripe_annihilation', 'Spin_torque']
							# _filter2 = (sname == 'Nucleation') and (notes.loc[i] == 'Bubble')
							if is_sign_given:
								_c, _alpha, edgecolors = fill_color, alpha, 'face'
								Ds_string = f"{Ds:.2f} pJ/m"
							else:
								_c, _alpha, edgecolors = 'lightgray', .5, 'black'
								Ds_string = f"|{Ds:.2f}| pJ/m"
							Ds_strings.append(Ds_string)
							if visualization_library == 'mpl':
								ax.scatter(x, y, c=_c, s=size**2, alpha=_alpha, marker=m, 
											edgecolors=edgecolors, linewidths=2)
						elif marker_type == 'fillings':
							color = "tab:%s" % fill_color
							fstyle, marker = fillstyle[sname]
							if visualization_library == 'mpl':
								ax.plot(x, y, markersize=size, alpha=alpha, fillstyle=fstyle, marker=marker,
										color=color, mew=2, **marker_style)
					else:
						if marker_type == 'shapes':
							if visualization_library == 'mpl':
								ax.scatter(x, y, c='black', s=size_factor, alpha=alpha, marker=chars[sname])
							else:
								Ds_string = f"0 pJ/m"
								Ds_strings.append(Ds_string)
						elif marker_type == 'fillings':
							if visualization_library == 'mpl':
								ax.plot(x, y, markersize=size_factor, marker='o', alpha=.2, fillstyle='full', color='black')
					n_data += 1
				n_elements.append(n_data)
				dic_n_elements[sname] = n_elements
			index.append("%s/%s/%s" % (bottom, fmat, top))
			df = pd.DataFrame(dic_n_elements, index=index)
			if df.empty:
				print("No data available")
				continue
			df.loc["Total"] = df.sum(axis=0)
			stack_key = self._stack_as_string(stack)
			self.n_experiments[stack_key] = df
			xaxis_label = 'bottom layer %s (nm)' % bottom
			yaxis_label = 'top layer %s (nm)' % top 
			title = self._get_title_topbottom(fmat, fmat_composition, bottom, top, select_D)
			if visualization_library == 'mpl':
				if set_axis is None:
					_max = np.max(_maxs) + 0.2
					ax.axis((0, _max, 0, _max))
				else:
					ax.axis(set_axis)
				#print("%i values" % n_data)
				xaxis_label_text_font_size = font_size
				yaxis_label_text_font_size = font_size
				title_text_font_size = font_size
				ax.set_xlabel(xaxis_label, fontsize=xaxis_label_text_font_size)
				ax.set_ylabel(yaxis_label, fontsize=yaxis_label_text_font_size)
				ax.set_title(title, fontsize=title_text_font_size)
				# Make the legends
				self._print_legends(sheetnames_with_data, marker_type, marker_colors, select_D, 
								size_factor, alpha, handle_color, zero_point=True)
				plt.show()
			elif visualization_library == 'bokeh':
				self.data = self.data.fillna("")
				items = ['Author', 'Title', 'Journal', 'Volume', 'Pages', 'Year']
				for item in items:
					self.data[item] = [self.papers.loc[ref][item] for ref in self.data.Ref]
				self.data['Author'] = self.get_authors_string(self.data['Author'])
				self.data['size'] = np.abs(self.data.Ds)**0.5 * size_factor
				self.data['alpha'] = [alpha if sign else 0.5 for sign in self.data.is_sign_given]
				self.data['marker'] = markers
				self.data['method'] = methods
				_line_colors, _fill_colors = [], []
				for sign, Ds in zip(self.data.is_sign_given, self.data.Ds):
					if sign:
						_c = marker_colors[np.sign(Ds)]
						_line_colors.append(_c)
						_fill_colors.append(_c)
					else:
						_line_colors.append('black')
						_fill_colors.append('lightgray')
				self.data['line_color'] = _line_colors
				self.data['fill_color'] = _fill_colors
				self.data['Ds_string'] = Ds_strings
				stack_string = self.get_stack_string(self.data)
				self.data['stack_string'] = stack_string

				tooltips = [('Stack', '@stack_string'),
							('D_s', '@Ds_string'), 
							('Method', '@method'),
							('Authors', '@Author'),
							('Tile', '@Title'),
							('Ref', '@Journal, @Volume, @Pages (@Year)'),
							('', ndots*'-')]
				fig = figure(plot_width=figsize[0], plot_height=figsize[1],
							 tooltips=tooltips, toolbar_location='above')
				fig.add_layout(LinearAxis(), 'right')
				fig.add_layout(LinearAxis(), 'above')
				fig.add_layout(Legend(), 'right')
				fig.scatter('tbottom', 'ttop', size='size', alpha='alpha', source=self.data, 
								line_color='line_color', fill_color='fill_color', marker='marker',
								legend_group='method')
				fig.xaxis.axis_label = xaxis_label
				fig.xaxis.axis_label_text_font_size = "25px"
				fig.yaxis.axis_label = yaxis_label
				fig.yaxis.axis_label_text_font_size = "22px"
				fig.yaxis.axis_label_text_font_style = "italic"
				#fig.yaxis.major_label_text_font_size = "25px"
				#fig.yaxis.major_label_text_font_style = "italic"
				if set_axis is not None:
					left, right, bottom, top = set_axis
					fig.x_range = Range1d(left, right)
					fig.y_range = Range1d(bottom, top)
				else:
					pass
					#fig.y_range = Range1d(1, len(elements)+1)
					#fig.xaxis.ticker = FixedTicker(ticks=list(range(1,len(elements)+1)))
				#size=20
				#ax.set_xlabel("%s layer thickness (nm)" % layer, size=20)
				#ax.set_ylabel("%s layer" % layer, size=20)
				# Make the legends
				#self._print_legends(sheetnames_with_data, marker_type, marker_colors, select_D, 
				#    size_factor, alpha, handle_color, zero_point=False)
				fig.legend.title = 'Methods'
				fig.legend.label_text_color = 'black'
				#items = []
				# for i in range(3):
				#     items += [(names[i],[fig.circle(i,i,color=colors[i],size=20)])]
				#     fig.add_layout(Legend(items=items),'right')
				
				title = title.replace("$","")
				fig.title.text = title.replace("_","")
				fig.title.text_font_size = "25px"
				t0, t1 = title.split("-")
				t1 = t1.replace("/", "_").strip()
				t1 = f"{t1}.html"
				#output_file(t1, title=t1)
		return fig


	def _h_leg(self, sheetnames_with_data):
		h_leg = (len(sheetnames_with_data) +1 ) * 0.06 + 0.025
		return h_leg
			
	def _get_title_fmat(self,fmat, fmat_composition, use_latex=False):
		title = ""
		if fmat == "CoFeB":
			fmat = ["Co", "Fe", "B"]
		if len(fmat_composition):
			fmat_composition = [fmat_composition[i:i+2] for i in range(0,6,2)]
			for f,c in zip(fmat,fmat_composition):
				title += r"%s_{%s}" % (f,c)
		else:
			title = "".join(fmat)
		return title

	def _get_title_stack(self, fmat, fmat_composition, fixed_layer, 
				fixed_layer_element, fixed_layer_thickness, select_D,use_latex=False):
		title = self._get_title_fmat(fmat, fmat_composition)
		if fixed_layer_thickness:
			_fixed_layer = "%s(%s)" % (fixed_layer_element, fixed_layer_thickness)
		else:
			_fixed_layer = fixed_layer_element
		if fixed_layer == 'bottom':
			q = (_fixed_layer, title, "X")
		else:
			q = ("X", title, _fixed_layer)
		title = r"$%s/%s/%s$" % q
		title = r"$%s$ - %s" % (select_D.replace("s","_s"), title)
		if use_latex:
			title = LatexLabel(
				text = title
				)
		
		return title

	def _get_title_topbottom(self, fmat, fmat_composition, bottom, top, select_D):
		title = self._get_title_fmat(fmat, fmat_composition)
		title = r"$%s/%s/%s$" % (bottom, title, top)
		title = r"$%s$ - %s" % (select_D.replace("s","_s"), title)
		return title

	def _stack_as_string(self, stack):
		return "__".join(["_".join(s) for s in stack])

	def make_latex_table(self, fm='CoFeB', fmat_composition='206020', bottom=None, top='MgO', 
						tabular_width=6.5, tabular_height=.2, bar_width=6):
		mainDir = "/home/gf/Documents/Papers_all/2020/DMIReview/New Figures"
		df = self.all_data_df
		if fmat_composition:
			fm_name = self._get_title_fmat(fm, fmat_composition)
		else:
			fm_name = fm
		if top and not bottom:
			_filter = (df.fm == fm) & (df.top == top)
			if fmat_composition:
				_filter = _filter & (df.composition == fmat_composition)
			out_table = "%s_%s.tex" % (fm, top)
			df = df[_filter]

			q = df[['bottom', 'tbottom', 'Ds']]
			qq = q.groupby(q.bottom).Ds.mean()
			qq.sort_values(inplace=True)
			elements = qq.index
		elif bottom and not top:
			_filter = (df.fm == fm) & (df.bottom == bottom)
			if composition:
				_filter = _filter & (df.composition == fmat_composition)
			out_table = "%s_%s.tex" % (bottom, fm)
			df = df[_filter]
			n_bottom = len(df)
			bottom_first = r"\multirow{%i}{*}{\centering %s}" % (len(df), bottom)
			q = df[['top', 'ttop', 'Ds']]
			qq = q.groupby(q.top).Ds.mean()
			elements = qq.index
		max_Ds = max(np.abs(df.Ds))
		print("Max Ds: %f" % max_Ds)
		out_table = os.path.join(mainDir, out_table)
		df.fillna("", inplace=True)
		w_white = bar_width//2
		edge_space = tabular_width - 2 * w_white
		max_Ds = (max_Ds//0.5 + 1) * 0.5
		s_white = r"\crule[white]{%.1fcm}{%.1fcm}" % (w_white, tabular_height)
		fm_name = fm_name.replace("_", "$_")
		fm_name = fm_name.replace("}", "}$")
		print(fm_name)
		
		with open(out_table, 'w') as t:
			print(r"\begin{table}", file=t)
			print(r"\begin{tabular}{c|L{%scm}|c|c|}" % tabular_width, file=t)
			print(r"\hline", file=t)
			print(r"\hline", file=t)
			s = r"\textbf{bottom NM} & \centering %s - $\mathbf{D_s}$" % fm_name
			s += r"& \textbf{top NM} & \textbf{Ref}\\" 
			print(s, file=t)
			print(r"\hline", file=t)
			print(r"\hline", file=t)
			for element in elements[:]:
				is_set_top = False
				if top:
					df1 = df[df.bottom==element]
					self.df1 = df1
					if len(df1) > 1:
						top_first = r"\multirow{%i}{*}{\hfil\centering %s}" % (len(df1), top)
					else:
						top_first = top
					#df1 = df1.sort_values(df1.Ds)
					for q in zip(df1.bottom, df1.tbottom, df1.bottom1, df1.tbottom1, 
									df1.Ds, df1.Method, df1.Ref):
						(b, tb, b1, tb1, Ds,meth, ref) = q
						try:
							s1 = "%s(%.1f)" % (b, tb) 
						except:
							print(b, tb)
						if b1 and tb1:
							s1 = "%s(%d)" % (b1, tb1) + s1
						#print(s1)
						w = Ds / max_Ds * w_white
						#print(Ds)
						if Ds > 0:
							s_crule = "%s" % (s_white)
							s_crule += r"\crule[%s]{%.1fcm}{%.1fcm}" % (meth, w, tabular_height)
							s_crule += r"\crule[white]{%.1fcm}{%.1fcm}" % (w_white-w, tabular_height)
						elif Ds < 0:
							w = np.abs(w)
							s_crule = r"\crule[white]{%.1fcm}{%.1fcm}" % (w_white-w, tabular_height)
							s_crule += r"\crule[%s]{%.1fcm}{%.1fcm}" % (meth, w, tabular_height)
							s_crule += "%s" % (s_white)
						if not is_set_top:
							s2 = top_first
							is_set_top = True
						else:
							s2 = "" 
						s = r"%s & %s & %s & \\" % (s1, s_crule, s2)
						#print(s)
						print(s, file=t)
				print(r"\hline", file=t)
			print(r"\hline", file=t)
			print(r"\end{tabular}", file=t)
			fm_out = fm
			if fmat_composition:
				fm_out += fmat_composition
			print(r"\caption{Summary table for %s}" % fm_out, file=t)
			print(r"\label{table:sumTable_%s}" % fm_out, file=t)
			print(r"\end{table}", file=t)
					
			
		return df


if __name__ == "__main__":
	# methods = ['DW_field_creep', 'DW_field_flow', 'DW_current', 'BLS']
	# #top, fmat, bottom = 'Pt', 'Co', 'Pt'
	# #bottom, fmat, top = 'W', 'CoFeB', 'MgO'
	# bottom, fmat, top, fmat_composition = 'Pt', 'CoFeB', 'MgO', "206020"
	# #bottom, fmat, top, fmat_composition = 'Pt', 'CoFeB', 'MgO', "404020"
	# #bottom, fmat, top = 'Ta', 'CoFeB', 'MgO'
	# #bottom, fmat, top = 'IrMn', 'CoFeB', 'MgO'
	# fmat, fmat_composition = 'CoFeB', "206020"
	# dmi = DMI(fmat, methods, fmat_composition=fmat_composition)
	#dmi.plotTopBottom("D", 100)
	#stack = (['Hf', 'Ta', 'TaN','W', 'IrMn'], ['MgO'])
	# Set some general variables. They can be redifined any time.
	methods = ['DW_field_creep', 'DW_field_flow', 'DW_current', 
				'Domain_pattern', 'Stripe_annihilation', 'Nucleation',
				'BLS', 'AESWS', 'Spin_orbit_torque']
	fsizeX = 5 # Size of the figure
	sfactor = 16 # Scale the size of the markers
	alpha = 0.5 # Set the transparency
	sel_D = "Ds"
	#stack = (['IrMn', 'Hf', 'Ta', 'TaN','W'], ['CoFeB', 'Co20Fe60B20'], ['MgO'])
	dmi = DMI(methods)
	#stack = (['IrMn', 'Hf', 'Pt', 'Ta', 'TaN','W'], ['CoFeB', 'Co20Fe60B20'], ['MgO'])
	#stack = (['IrMn', 'Hf', 'Pt', 'Ta', 'TaN','W(2-3)'], ['Co20Fe60B20'], ['MgO'])
	#dmi.plotStack(stack, "Ds", sfactor, alpha, fsizeX, 'fillings')
	#stack = (['Pt'], ['Co'], ['Pt'])
	stack = (['IrMn', 'Hf', 'Pt', 'Ta', 'TaN','W'], ['Co20Fe60B20'], ['MgO'])
	stack = (['IrMn', 'Hf', 'Pt', 'Ta', 'TaN','W'], ['CoFeB'], ['MgO'])
	#dmi.plotStack(stack, "Ds", sfactor, alpha, fsizeX, 'shapes',visualization_library='bokeh')
	#dmi.print_experiments()
	fig25 = dmi.plotStack(stack, sel_D, size_factor=30, figsizeX=(800,600), alpha=.5, 
					 marker_type='shapes', marker_colors=colors, handle_color='gray',
					 set_axis=(-0.2,8,None,None), visualization_library='bokeh')
	#out = dmi.make_latex_table(fm='CoFeB', fmat_composition='206020', bottom=None, top='MgO',max_Ds=1.7)
	#out = dmi.make_latex_table(fm='CoFeB', fmat_composition=None, bottom=None, top='MgO')