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
		var pix1, pix2, pix = ((canvas.width * canvas.height) << 2);
		while (pix > 0)
			data[pix -= 4] = data[pix1 = pix + 1] = data[pix2 = pix + 2] = (data[pix] * 0.3 + data[pix1] * 0.59 + data[pix2] * 0.11);
		ctx.putImageData(imageData, 0, 0);
		return canvas;
	},

	array_group : function (seq, gfunc) {
		var i, j;
		var seq_length = seq.length;
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
			this.shared.scale_upto = ((Math.log(Math.min(params.canvas.width / params.cascade.width, params.canvas.height / params.cascade.height)) / Math.log(this.shared.scale)) | 0);
			var i;
			var this_shared_cascade_stage_classifier = this.shared.cascade.stage_classifier;
			var this_shared_cascade_stage_classifier_length = this_shared_cascade_stage_classifier.length;
			for (i = 0; i < this_shared_cascade_stage_classifier_length; i++)
				this_shared_cascade_stage_classifier[i].orig_feature = this_shared_cascade_stage_classifier[i].feature;
		}
		function pre(worker_num) {
			var canvas = this.shared.canvas;
			var interval = this.shared.interval;
			var scale = this.shared.scale;
			var next = this.shared.next;
			var scale_upto = this.shared.scale_upto;
			var scale_upto_next_1 = scale_upto + (next << 1);
			var pyr = new Array((scale_upto + (next << 1)) << 2);
			var ret = new Array((scale_upto + (next << 1)) << 2);
			pyr[0] = canvas;
			ret[0] = { "width" : pyr[0].width,
					   "height" : pyr[0].height,
					   "data" : pyr[0].getContext("2d").getImageData(0, 0, pyr[0].width, pyr[0].height).data };
			var i;
			for (i = 1; i <= interval; i++) {
				pyr[i << 2] = document.createElement("canvas");
				pyr[i << 2].width = ((pyr[0].width / Math.pow(scale, i)) | 0);
				pyr[i << 2].height = ((pyr[0].height / Math.pow(scale, i) | 0));
				pyr[i << 2].getContext("2d").drawImage(pyr[0], 0, 0, pyr[0].width, pyr[0].height, 0, 0, pyr[i << 2].width, pyr[i << 2].height);
				ret[i << 2] = { "width" : pyr[i << 2].width,
							   "height" : pyr[i << 2].height,
							   "data" : pyr[i << 2].getContext("2d").getImageData(0, 0, pyr[i << 2].width, pyr[i << 2].height).data };
			}
			for (i = next; i < scale_upto_next_1; i++) {
				pyr[i << 2] = document.createElement("canvas");
				pyr[i << 2].width = ((pyr[(i << 2) - (next << 2)].width / 2) | 0);
				pyr[i << 2].height = ((pyr[(i << 2) - (next << 2)].height / 2) | 0);
				pyr[i << 2].getContext("2d").drawImage(pyr[(i << 2) - (next << 2)], 0, 0, pyr[(i << 2) - (next << 2)].width, pyr[(i << 2) - (next << 2)].height, 0, 0, pyr[i << 2].width, pyr[i << 2].height);
				ret[i << 2] = { "width" : pyr[i << 2].width,
							   "height" : pyr[i << 2].height,
							   "data" : pyr[i << 2].getContext("2d").getImageData(0, 0, pyr[i << 2].width, pyr[i << 2].height).data };
			}
			for (i = (next << 1); i < scale_upto_next_1; i++) {
				pyr[(i << 2) + 1] = document.createElement("canvas");
				pyr[(i << 2) + 1].width = ((pyr[(i << 2) - (next << 2)].width / 2) | 0);
				pyr[(i << 2) + 1].height = ((pyr[(i << 2) - (next << 2)].height / 2) | 0);
				pyr[(i << 2) + 1].getContext("2d").drawImage(pyr[(i << 2) - (next << 2)], 1, 0, pyr[(i << 2) - (next << 2)].width - 1, pyr[(i << 2) - (next << 2)].height, 0, 0, pyr[(i << 2) + 1].width - 2, pyr[(i << 2) + 1].height);
				ret[(i << 2) + 1] = { "width" : pyr[(i << 2) + 1].width,
								   "height" : pyr[(i << 2) + 1].height,
								   "data" : pyr[(i << 2) + 1].getContext("2d").getImageData(0, 0, pyr[(i << 2) + 1].width, pyr[(i << 2) + 1].height).data };
				pyr[(i << 2) + 2] = document.createElement("canvas");
				pyr[(i << 2) + 2].width = ((pyr[(i << 2) - (next << 2)].width / 2) | 0);
				pyr[(i << 2) + 2].height = ((pyr[(i << 2) - (next << 2)].height / 2) | 0);
				pyr[(i << 2) + 2].getContext("2d").drawImage(pyr[(i << 2) - (next << 2)], 0, 1, pyr[(i << 2) - (next << 2)].width, pyr[(i << 2) - (next << 2)].height - 1, 0, 0, pyr[(i << 2) + 2].width, pyr[(i << 2) + 2].height - 2);
				ret[(i << 2) + 2] = { "width" : pyr[(i << 2) + 2].width,
								   "height" : pyr[(i << 2) + 2].height,
								   "data" : pyr[(i << 2) + 2].getContext("2d").getImageData(0, 0, pyr[(i << 2) + 2].width, pyr[(i << 2) + 2].height).data };
				pyr[(i << 2) + 3] = document.createElement("canvas");
				pyr[(i << 2) + 3].width = ((pyr[(i << 2) - (next << 2)].width / 2) | 0);
				pyr[(i << 2) + 3].height = ((pyr[(i << 2) - (next << 2)].height / 2) | 0);
				pyr[(i << 2) + 3].getContext("2d").drawImage(pyr[(i << 2) - (next << 2)], 1, 1, pyr[(i << 2) - (next << 2)].width - 1, pyr[(i << 2) - (next << 2)].height - 1, 0, 0, pyr[(i << 2) + 3].width - 2, pyr[(i << 2) + 3].height - 2);
				ret[(i << 2) + 3] = { "width" : pyr[(i << 2) + 3].width,
								   "height" : pyr[(i << 2) + 3].height,
								   "data" : pyr[(i << 2) + 3].getContext("2d").getImageData(0, 0, pyr[(i << 2) + 3].width, pyr[(i << 2) + 3].height).data };
			}
			return [ret];
		};

		function core(pyr, id, worker_num) {
			var cascade = this.shared.cascade;
			var cascade_stage_classifier = cascade.stage_classifier;
			var cascade_stage_classifier_length = cascade_stage_classifier.length;
			var interval = this.shared.interval;
			var scale = this.shared.scale;
			var next = this.shared.next;
			var scale_upto = this.shared.scale_upto;
			var i, j, k, x, y, q;
			var scale_x = 1, scale_y = 1;
			var dx = [0, 1, 0, 1];
			var dy = [0, 0, 1, 1];
			var seq = [];
			for (i = 0; i < scale_upto; i++) {
				var qw = pyr[(i << 2) + (next << 3)].width - ((cascade.width / 4) | 0);
				var qh = pyr[(i << 2) + (next << 3)].height - ((cascade.height / 4) | 0);
				var step = [(pyr[i << 2].width << 2), (pyr[(i << 2) + (next << 2)].width << 2), (pyr[(i << 2) + (next << 3)].width << 2)];
				var paddings = [(pyr[i << 2].width << 4) - (qw << 4),
								(pyr[(i << 2) + (next << 2)].width << 3) - (qw << 3),
								(pyr[(i << 2) + (next << 3)].width << 2) - (qw << 2)];
				for (j = 0; j < cascade_stage_classifier_length; j++) {
					var cascade_stage_classifier_j = cascade_stage_classifier[j];
					var cascade_stage_classifier_j_count = cascade_stage_classifier_j.count;
					var orig_feature = cascade_stage_classifier_j.orig_feature;
					var feature = cascade_stage_classifier_j.feature = new Array(cascade_stage_classifier_j_count);
					for (k = 0; k < cascade_stage_classifier_j_count; k++) {
						var orig_feature_k = orig_feature[k];
						var orig_feature_k_size = orig_feature_k.size;
						var orig_feature_k_px = orig_feature_k.px;
						var orig_feature_k_py = orig_feature_k.py;
						var orig_feature_k_pz = orig_feature_k.pz;
						var orig_feature_k_nx = orig_feature_k.nx;
						var orig_feature_k_ny = orig_feature_k.ny;
						var orig_feature_k_nz = orig_feature_k.nz;
						feature[k] = {"size" : orig_feature_k_size,
									  "px" : new Array(orig_feature_k_size),
									  "pz" : new Array(orig_feature_k_size),
									  "nx" : new Array(orig_feature_k_size),
									  "nz" : new Array(orig_feature_k_size)};
						var feature_k = feature[k];
						for (q = 0; q < orig_feature_k_size; q++) {
							feature_k.px[q] = (orig_feature_k_px[q] << 2) + orig_feature_k_py[q] * step[orig_feature_k_pz[q]];
							feature_k.pz[q] = orig_feature_k_pz[q];
							feature_k.nx[q] = (orig_feature_k_nx[q] << 2) + orig_feature_k_ny[q] * step[orig_feature_k_nz[q]];
							feature_k.nz[q] = orig_feature_k_nz[q];
						}
					}
				}
				for (q = 0; q < 4; q++) {
					var u8 = [pyr[i << 2].data, pyr[(i << 2) + (next << 2)].data, pyr[(i << 2) + (next << 3) + q].data];
					var u8o = [(dx[q] << 3) + dy[q] * (pyr[i << 2].width << 3), (dx[q] << 2) + dy[q] * (pyr[(i << 2) + (next << 2)].width << 2), 0];
					for (y = 0; y < qh; y++) {
						for (x = 0; x < qw; x++) {
							var sum = 0;
							var flag = true;
							for (j = 0; j < cascade_stage_classifier_length; j++) {
								var cascade_stage_classifier_j = cascade_stage_classifier[j];
								var cascade_stage_classifier_j_count = cascade_stage_classifier_j.count;
								sum = 0;
								var alpha = cascade_stage_classifier_j.alpha;
								var feature = cascade_stage_classifier_j.feature;
								for (k = 0; k < cascade_stage_classifier_j_count; k++) {
									var feature_k = feature[k];
									var feature_k_px = feature_k.px;
									var feature_k_pz = feature_k.pz;
									var feature_k_nx = feature_k.nx;
									var feature_k_nz = feature_k.nz;
									var p, pmin = u8[feature_k_pz[0]][u8o[feature_k_pz[0]] + feature_k_px[0]];
									var n, nmax = u8[feature_k_nz[0]][u8o[feature_k_nz[0]] + feature_k_nx[0]];
									if (pmin <= nmax) {
										sum += alpha[k << 1];
									} else {
										var f, shortcut = true;
										var feature_k_size = feature_k.size;
										for (f = 0; f < feature_k_size; f++) {
											if (feature_k_pz[f] >= 0) {
												p = u8[feature_k_pz[f]][u8o[feature_k_pz[f]] + feature_k_px[f]];
												if (p < pmin) {
													if (p <= nmax) {
														shortcut = false;
														break;
													}
													pmin = p;
												}
											}
											if (feature_k_nz[f] >= 0) {
												n = u8[feature_k_nz[f]][u8o[feature_k_nz[f]] + feature_k_nx[f]];
												if (n > nmax) {
													if (pmin <= n) {
														shortcut = false;
														break;
													}
													nmax = n;
												}
											}
										}
										sum += (shortcut) ? alpha[(k << 1) + 1] : alpha[k << 1];
									}
								}
								if (sum < cascade_stage_classifier_j.threshold) {
									flag = false;
									break;
								}
							}
							if (flag) {
								seq.push({"x" : ((x << 2) + (dx[q] << 1)) * scale_x,
										  "y" : ((y << 2) + (dy[q] << 1)) * scale_y,
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
			var cascade_stage_classifier = cascade.stage_classifier;
			var cascade_stage_classifier_length = cascade_stage_classifier.length;
			var interval = this.shared.interval;
			var scale = this.shared.scale;
			var next = this.shared.next;
			var scale_upto = this.shared.scale_upto;
			var i, j;
			for (i = 0; i < cascade_stage_classifier_length; i++)
				cascade_stage_classifier[i].feature = cascade_stage_classifier[i].orig_feature;
			seq = seq[0];
			if (!(min_neighbors > 0))
				return seq;
			else {
				var result = ccv.array_group(seq, function (r1, r2) {
					var distance = ((r1.width * 0.25 + 0.5) | 0);

					return r2.x <= r1.x + distance &&
						   r2.x >= r1.x - distance &&
						   r2.y <= r1.y + distance &&
						   r2.y >= r1.y - distance &&
						   r2.width <= ((r1.width * 1.5 + 0.5) | 0) &&
						   ((r2.width * 1.5 + 0.5) | 0) >= r1.width;
				});
				var ncomp = result.cat;
				var idx_seq = result.index;
				var comps = new Array(ncomp + 1);
				for (i = 0; i < comps.length; i++)
					comps[i] = {"neighbors" : 0,
								"x" : 0,
								"y" : 0,
								"width" : 0,
								"height" : 0,
								"confidence" : 0};

				// count number of neighbors
				var seq_length = seq.length;
				for(i = 0; i < seq_length; i++)
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
				// calculate average bounding box
				for(i = 0; i < ncomp; i++)
				{
					var n = comps[i].neighbors;
					if (n >= min_neighbors)
						seq2.push({"x" : (comps[i].x * 2 + n) / (2 * n),
								   "y" : (comps[i].y * 2 + n) / (2 * n),
								   "width" : (comps[i].width * 2 + n) / (2 * n),
								   "height" : (comps[i].height * 2 + n) / (2 * n),
								   "neighbors" : comps[i].neighbors,
								   "confidence" : comps[i].confidence});
				}

				var result_seq = [];
				// filter out small face rectangles inside large face rectangles
				for(i = 0; i < seq2.length; i++)
				{
					var r1 = seq2[i];
					var flag = true;
					for(j = 0; j < seq2.length; j++)
					{
						var r2 = seq2[j];
						var distance = ((r2.width * 0.25 + 0.5) | 0);

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
