if (parallable === undefined) {
	var parallable = function (file, funct) {
		parallable.core[funct.toString()] = funct().core;
		return function () {
			var i;
			var async, worker_num, params;
			if (arguments.length > 1) {
				async = arguments[arguments.length - 2];
				worker_num = arguments[arguments.length - 1];
				params = new Array(arguments.length - 2);
				for (i = 0; i < arguments.length - 2; i++)
					params[i] = arguments[i];
			} else {
				async = arguments[0].async;
				worker_num = arguments[0].worker;
				params = arguments[0];
				delete params["async"];
				delete params["worker"];
				params = [params];
			}
			var scope = { "shared" : {} };
			var ctrl = funct.apply(scope, params);
			if (async) {
				return function (complete, error) {
					var executed = 0;
					var outputs = new Array(worker_num);
					var inputs = ctrl.pre.apply(scope, [worker_num]);
					/* sanitize scope shared because for Chrome/WebKit, worker only support JSONable data */
					for (i in scope.shared)
						/* delete function, if any */
						if (typeof scope.shared[i] == "function")
							delete scope.shared[i];
						/* delete DOM object, if any */
						else if (scope.shared[i].tagName !== undefined)
							delete scope.shared[i];
					for (i = 0; i < worker_num; i++) {
						var worker = new Worker(file);
						worker.onmessage = (function (i) {
							return function (event) {
								outputs[i] = (typeof event.data == "string") ? JSON.parse(event.data) : event.data;
								executed++;
								if (executed == worker_num)
									complete(ctrl.post.apply(scope, [outputs]));
							}
						})(i);
						var msg = { "input" : inputs[i],
									"name" : funct.toString(),
									"shared" : scope.shared,
									"id" : i,
									"worker" : params.worker_num };
						try {
							worker.postMessage(msg);
						} catch (e) {
							worker.postMessage(JSON.stringify(msg));
						}
					}
				}
			} else {
				return ctrl.post.apply(scope, [[ctrl.core.apply(scope, [ctrl.pre.apply(scope, [1])[0], 0, 1])]]);
			}
		}
	};
	parallable.core = {};
}

function get_named_arguments(params, names) {
	if (params.length > 1) {
		var new_params = {};
		for (var i = 0; i < names.length; i++)
			new_params[names[i]] = params[i];
		return new_params;
	} else if (params.length == 1) {
		return params[0];
	} else {
		return {};
	}
}

var ccv = {
	pre : function (image) {
		if (image.tagName.toLowerCase() == "img") {
			var canvas = document.createElement("canvas");
			document.body.appendChild(image);
			canvas.width = image.offsetWidth;
			canvas.style.width = image.offsetWidth.toString() + "px";
			canvas.height = image.offsetHeight;
			canvas.style.height = image.offsetHeight.toString() + "px";
			document.body.removeChild(image);
			var ctx = canvas.getContext("2d");
			ctx.drawImage(image, 0, 0);
			return canvas;
		}
		return image;
	},

	grayscale : function (canvas) {
		var ctx = canvas.getContext("2d");
		var imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
		var data = imageData.data;
		var pix1, pix2, pix = (canvas.width * canvas.height) << 2; // * 4 converted to bitwise
		while (pix > 0)
			data[pix -= 4] = data[pix1 = pix + 1] = data[pix2 = pix + 2] = (data[pix] * 0.3 + data[pix1] * 0.59 + data[pix2] * 0.11);
		ctx.putImageData(imageData, 0, 0);
		return canvas;
	},

	array_group : function (seq, gfunc) {
		// cache loop limits (seq.length)
		var i, j, seq_length = seq.length;
		var node = new Array(seq_length);
		for (i = 0; i < seq_length; i++)
			node[i] = {"parent" : -1,
						 "element" : seq[i],
						 "rank" : 0};
		for (i = 0; i < seq_length; i++) {
			if (!node[i].element)
				continue;
			var root = i;
			while (node[root].parent != -1)
				root = node[root].parent;
			for (j = 0; j < seq_length; j++) {
				if( i != j && node[j].element && gfunc(node[i].element, node[j].element)) {
					var root2 = j;

					while (node[root2].parent != -1)
						root2 = node[root2].parent;

					if(root2 != root) {
						if(node[root].rank > node[root2].rank)
							node[root2].parent = root;
						else {
							node[root].parent = root2;
							if (node[root].rank == node[root2].rank)
							node[root2].rank++;
							root = root2;
						}

						/* compress path from node2 to the root: */
						var temp, node2 = j;
						while (node[node2].parent != -1) {
							temp = node2;
							node2 = node[node2].parent;
							node[temp].parent = root;
						}

						/* compress path from node to the root: */
						node2 = i;
						while (node[node2].parent != -1) {
							temp = node2;
							node2 = node[node2].parent;
							node[temp].parent = root;
						}
					}
				}
			}
		}
		var idx = new Array(seq_length);
		var class_idx = 0;
		for(i = 0; i < seq_length; i++) {
			j = -1;
			var node1 = i;
			if(node[node1].element) {
				while (node[node1].parent != -1)
					node1 = node[node1].parent;
				if(node[node1].rank >= 0)
					node[node1].rank = ~class_idx++;
				j = ~node[node1].rank;
			}
			idx[i] = j;
		}
		return {"index" : idx, "cat" : class_idx};
	},

	detect_objects : parallable("ccv.js", function (canvas, cascade, interval, min_neighbors) {
		if (this.shared !== undefined) {
			var params = get_named_arguments(arguments, ["canvas", "cascade", "interval", "min_neighbors"]);
			this.shared.canvas = params.canvas;
			this.shared.interval = params.interval;
			this.shared.min_neighbors = params.min_neighbors;
			this.shared.cascade = params.cascade;
			this.shared.scale = Math.pow(2, 1 / (params.interval + 1));
			this.shared.next = params.interval + 1;
			this.shared.scale_upto = Math.floor(Math.log(Math.min(params.canvas.width / params.cascade.width, params.canvas.height / params.cascade.height)) / Math.log(this.shared.scale));
			// cache loop limit, cache object reference.
			var i, i_length = this.shared.cascade.stage_classifier.length, this_stage_cascade_stage_classifier = this.stage.cascade.stage_classifier;
			for (i = 0; i < i_length; i++)
				this_stage_cascade_stage_classifier[i].orig_feature = this_stage_cascade_stage_classifier[i].feature;
		}
		function pre(worker_num) {
			var canvas = this.shared.canvas;
			var interval = this.shared.interval;
			var scale = this.shared.scale;
			var next = this.shared.next;
			var scale_upto = this.shared.scale_upto;
			var arr_size = (scale_upto + next << 1) << 2; // * 2, * 4 converted to bitwise
			var pyr = new Array(arr_size);
			var ret = new Array(arr_size);
			pyr[0] = canvas;
			var i, i_limit, index, index0, index1, index2, index3, pyr_item0 = pyr[0], pyr_item1, pyr_item2, pyr_item3, w0 = pyr_item0.width, w1, h0 = pyr_item0.height, h1;
			ret[0] = { "width" : pyr[0].width,
						 "height" : pyr[0].height,
						 "data" : pyr[0].getContext("2d").getImageData(0, 0, pyr[0].width, pyr[0].height).data };
			for (i = 1; i <= interval; i++) {
				index = i << 2;
				w1 = Math.floor(w0 / Math.pow(scale, i));
				h1 = Math.floor(h0 / Math.pow(scale, i));
				// cache reference to pyr[index]
				pyr_item1 = pyr[index] = document.createElement("canvas");
				pyr_item1.width = w1;
				pyr_item1.height = h1;
				pyr_item1.getContext("2d").drawImage(pyr_item0, 0, 0, w0, h0, 0, 0, w1, h1);
				ret[index] = { "width" : w1,
								 "height" : h1,
								 "data" : pyr_item1.getContext("2d").getImageData(0, 0, w1, h1).data };
			}
			for (i = next, i_limit = scale_upto + next << 1; i < i_limit; i++) {
				index = i << 2;
				pyr_item0 = pyr[index - next << 2];
				w0 = pyr_item0.width;
				h0 = pyr_item0.height;
				w1 = w0 >> 1;
				h1 = h0 >> 1;
				// cache reference to pyr[index]
				pyr_item1 = pyr[index] = document.createElement("canvas");
				pyr_item1.width = w1;
				pyr_item1.height = h1;
				pyr_item1.getContext("2d").drawImage(pyr_item0, 0, 0, w0, h0, 0, 0, w1, h1);
				ret[index] = { "width" : w1,
								 "height" : h1,
								 "data" : pyr_item1.getContext("2d").getImageData(0, 0, w1, h1).data };
			}
			for (i = next << 1, i_limit = scale_upto + next << 1; i < i_limit; i++) {
				index0 = i << 2; // i * 4
				index1 = index0 + 1;
				index2 = index1 + 1;
				index3 = index2 + 1;
				pyr_item0 = pyr[index0 - next << 2];
				w0 = pyr_item0.width;
				h0 = pyr_item0.height;
				w1 = w0 >> 1;
				h1 = h0 >> 1;
				pyr_item1 = pyr[index1] = document.createElement("canvas");
				pyr_item1.width = w1;
				pyr_item1.height = h1;
				pyr_item1.getContext("2d").drawImage(pyr_item0, 1, 0, w0 - 1, h0, 0, 0, w1 - 2, h1);
				ret[index1] = { "width" : w1,
									 "height" : h1,
									 "data" : pyr_item1.getContext("2d").getImageData(0, 0, w1, h1).data };
				pyr_item2 = pyr[index2] = document.createElement("canvas");
				pyr_item2.width = w1;
				pyr_item2.height = h1;
				pyr_item2.getContext("2d").drawImage(pyr_item0, 0, 1, w0, h0 - 1, 0, 0, w1, h1 - 2);
				ret[index2] = { "width" : w1,
									 "height" : h1,
									 "data" : pyr_item2.getContext("2d").getImageData(0, 0, w1, h1).data };
				pyr_item3 = pyr[index3] = document.createElement("canvas");
				pyr_item3.width = w1;
				pyr_item3.height = h1;
				pyr_item3.getContext("2d").drawImage(pyr_item0, 1, 1, w0 - 1, h0 - 1, 0, 0, w1 - 2, h1 - 2);
				ret[index3] = { "width" : w1,
									 "height" : h1,
									 "data" : pyr_item3.getContext("2d").getImageData(0, 0, w1, h1).data };
			}
			return [ret];
		};

		function core(pyr, id, worker_num) {
			var shared = this.shared;
			var cascade = shared.cascade;
			var interval = shared.interval;
			var scale = shared.scale;
			var next = shared.next;
			var scale_upto = shared.scale_upto;
			var i, j, k, x, y, q, j_limit, k_limit;
			var scale_x = 1, scale_y = 1;
			var dx = [0, 1, 0, 1];
			var dy = [0, 0, 1, 1];
			var seq = [];
			for (i = 0; i < scale_upto; i++) {
				var index = i << 2;
				var pyr_item0 = pyr[index + next << 3];
				var pyr_item1 = pyr[index];
				var pyr_item2 = pyr[index + next << 2];
				var w0 = pyr_item0.width;
				var h0 = pyr_item0.height;
				var qw = w0 - cascade.width >> 2;
				var qh = h0 - cascade.height >> 2;
				var step = [pyr_item1.width << 2, pyr_item2.width << 2, w0 << 2];
				var paddings = [pyr_item1.width << 4 - qw << 4,
								pyr_item2.width << 3 - qw << 3,
								w0 << 2 - qw << 2];
				for (j = 0, j_limit = cascade.stage_classifier.length; j < j_limit; j++) {
					var orig_feature = cascade.stage_classifier[j].orig_feature;
					var feature = cascade.stage_classifier[j].feature = new Array(cascade.stage_classifier[j].count);
					for (k = 0, k_limit = cascade.stage_classifier[j].count; k < k_limit; k++) {
						var size = orig_feature[k].size;
						feature[k] = {"size" : size,
										"px" : new Array(size),
										"pz" : new Array(size),
										"nx" : new Array(size),
										"nz" : new Array(size)};
						for (q = 0; q < size; q++) {
							feature[k].px[q] = orig_feature[k].px[q] << 2 + orig_feature[k].py[q] * step[orig_feature[k].pz[q]];
							feature[k].pz[q] = orig_feature[k].pz[q];
							feature[k].nx[q] = orig_feature[k].nx[q] << 2 + orig_feature[k].ny[q] * step[orig_feature[k].nz[q]];
							feature[k].nz[q] = orig_feature[k].nz[q];
						}
					}
				}
				for (q = 0; q < 4; q++) {
					var index = i << 2;
					var u8 = [pyr[index].data, pyr[index + next << 2].data, pyr[index + next << 3 + q].data];
					var u8o = [dx[q] * 8 + dy[q] * pyr[index].width << 3, dx[q] * 4 + dy[q] * pyr[index + next << 2].width << 2, 0];
					for (y = 0; y < qh; y++) {
						for (x = 0; x < qw; x++) {
							var sum = 0;
							var flag = true;
							for (j = 0, j_limit = cascade.stage_classifier.length; j < j_limit; j++) {
								sum = 0;
								var alpha = cascade.stage_classifier[j].alpha;
								var feature = cascade.stage_classifier[j].feature;
								for (k = 0, k_limit = cascade.stage_classifier[j].count; k < k_limit; k++) {
									var feature_k = feature[k];
									var p, pmin = u8[feature_k.pz[0]][u8o[feature_k.pz[0]] + feature_k.px[0]];
									var n, nmax = u8[feature_k.nz[0]][u8o[feature_k.nz[0]] + feature_k.nx[0]];
									if (pmin <= nmax) {
										sum += alpha[k * 2];
									} else {
										var f, shortcut = true;
										for (f = 0; f < feature_k.size; f++) {
											if (feature_k.pz[f] >= 0) {
												p = u8[feature_k.pz[f]][u8o[feature_k.pz[f]] + feature_k.px[f]];
												if (p < pmin) {
													if (p <= nmax) {
														shortcut = false;
														break;
													}
													pmin = p;
												}
											}
											if (feature_k.nz[f] >= 0) {
												n = u8[feature_k.nz[f]][u8o[feature_k.nz[f]] + feature_k.nx[f]];
												if (n > nmax) {
													if (pmin <= n) {
														shortcut = false;
														break;
													}
													nmax = n;
												}
											}
										}
										sum += (shortcut) ? alpha[k * 2 + 1] : alpha[k * 2];
									}
								}
								if (sum < cascade.stage_classifier[j].threshold) {
									flag = false;
									break;
								}
							}
							if (flag) {
								seq.push({"x" : (x * 4 + dx[q] * 2) * scale_x,
											"y" : (y * 4 + dy[q] * 2) * scale_y,
											"width" : cascade.width * scale_x,
											"height" : cascade.height * scale_y,
											"neighbor" : 1,
											"confidence" : sum});
							}
							u8o[0] += 16;
							u8o[1] += 8;
							u8o[2] += 4;
						}
						u8o[0] += paddings[0];
						u8o[1] += paddings[1];
						u8o[2] += paddings[2];
					}
				}
				scale_x *= scale;
				scale_y *= scale;
			}
			return seq;
		};

		function post(seq) {
			var min_neighbors = this.shared.min_neighbors;
			var cascade = this.shared.cascade;
			var interval = this.shared.interval;
			var scale = this.shared.scale;
			var next = this.shared.next;
			var scale_upto = this.shared.scale_upto;
			var i, j, i_limit;
			for (i = 0, i_limit = cascade.stage_classifier.length; i < i_limit; i++)
				cascade.stage_classifier[i].feature = cascade.stage_classifier[i].orig_feature;
			seq = seq[0];
			if (!(min_neighbors > 0))
				return seq;
			else {
				var result = ccv.array_group(seq, function (r1, r2) {
					var distance = Math.floor(r1.width * 0.25 + 0.5);

					return r2.x <= r1.x + distance &&
							 r2.x >= r1.x - distance &&
							 r2.y <= r1.y + distance &&
							 r2.y >= r1.y - distance &&
							 r2.width <= Math.floor(r1.width * 1.5 + 0.5) &&
							 Math.floor(r2.width * 1.5 + 0.5) >= r1.width;
				});
				var ncomp = result.cat;
				var idx_seq = result.index;
				var comps = new Array(ncomp + 1);
				for (i = 0, i_limit = comps.length; i < i_limit; i++)
					comps[i] = {"neighbors" : 0,
								"x" : 0,
								"y" : 0,
								"width" : 0,
								"height" : 0,
								"confidence" : 0};

				// count number of neighbors
				for(i = 0, i_limit = seq.length; i < i_limit; i++)
				{
					var r1 = seq[i];
					var idx = idx_seq[i];

					if (comps[idx].neighbors == 0)
						comps[idx].confidence = r1.confidence;

					++comps[idx].neighbors;

					comps[idx].x += r1.x;
					comps[idx].y += r1.y;
					comps[idx].width += r1.width;
					comps[idx].height += r1.height;
					comps[idx].confidence = Math.max(comps[idx].confidence, r1.confidence);
				}

				var seq2 = [];
				var comp;
				// calculate average bounding box
				for(i = 0; i < ncomp; i++)
				{
					var n = comps[i].neighbors;
					if (n >= min_neighbors){
						comp = comps[i];
						seq2.push({"x" : (comp.x * 2 + n) / (2 * n),
									 "y" : (comp.y * 2 + n) / (2 * n),
									 "width" : (comp.width * 2 + n) / (2 * n),
									 "height" : (comp.height * 2 + n) / (2 * n),
									 "neighbors" : comp.neighbors,
									 "confidence" : comp.confidence});
					 }
				}

				var result_seq = [];
				var seq2_length = seq2.length;
				// filter out small face rectangles inside large face rectangles
				for(i = 0; i < seq2_length; i++)
				{
					var r1 = seq2[i];
					var flag = true;
					for(j = 0; j < seq2_length; j++)
					{
						var r2 = seq2[j];
						var distance = Math.floor(r2.width * 0.25 + 0.5);

						if(i != j &&
							 r1.x >= r2.x - distance &&
							 r1.y >= r2.y - distance &&
							 r1.x + r1.width <= r2.x + r2.width + distance &&
							 r1.y + r1.height <= r2.y + r2.height + distance &&
							 (r2.neighbors > Math.max(3, r1.neighbors) || r1.neighbors < 3))
						{
							flag = false;
							break;
						}
					}

					if(flag)
						result_seq.push(r1);
				}
				return result_seq;
			}
		};
		return { "pre" : pre, "core" : core, "post" : post };
	})
}

onmessage = function (event) {
	var data = (typeof event.data == "string") ? JSON.parse(event.data) : event.data;
	var scope = { "shared" : data.shared };
	var result = parallable.core[data.name].apply(scope, [data.input, data.id, data.worker]);
	try {
		postMessage(result);
	} catch (e) {
		postMessage(JSON.stringify(result));
	}
}
