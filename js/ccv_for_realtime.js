var ccv = {

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

	detect : function(params) {
		return this.post(params, this.pre_and_core(params));
	},

	pre_and_core : function(params) {

		var canvas = params.canvas;
		var cascade = params.cascade;
		var interval = params.interval;
		var scale = Math.pow(2, 1 / (interval + 1));
		var next = interval + 1;

		var i, j, k, x, y, q;

		var canvas_width = canvas.width;
		var canvas_height = canvas.height;
		var scale_upto = ((Math.log(Math.min(canvas_width / cascade.width, canvas_height / cascade.height)) / Math.log(scale)) | 0);

		var scale_upto_next_1 = scale_upto + (next << 1);
		var pyr_buf = new Array((scale_upto + (next << 1)) << 2);
		var pyr = new Array((scale_upto + (next << 1)) << 2);
		pyr_buf[0] = canvas;
		pyr[0] = { "width" : pyr_buf[0].width,
				   "height" : pyr_buf[0].height,
				   "data" : pyr_buf[0].getContext("2d").getImageData(0, 0, pyr_buf[0].width, pyr_buf[0].height).data };
		for (i = 1; i <= interval; i++) {
			pyr_buf[i << 2] = document.createElement("canvas");
			pyr_buf[i << 2].width = ((pyr_buf[0].width / Math.pow(scale, i)) | 0);
			pyr_buf[i << 2].height = ((pyr_buf[0].height / Math.pow(scale, i) | 0));
			pyr_buf[i << 2].getContext("2d").drawImage(pyr_buf[0], 0, 0, pyr_buf[0].width, pyr_buf[0].height, 0, 0, pyr_buf[i << 2].width, pyr_buf[i << 2].height);
			pyr[i << 2] = { "width" : pyr_buf[i << 2].width,
						    "height" : pyr_buf[i << 2].height,
						    "data" : pyr_buf[i << 2].getContext("2d").getImageData(0, 0, pyr_buf[i << 2].width, pyr_buf[i << 2].height).data };
		}
		for (i = next; i < scale_upto_next_1; i++) {
			pyr_buf[i << 2] = document.createElement("canvas");
			pyr_buf[i << 2].width = ((pyr_buf[(i << 2) - (next << 2)].width / 2) | 0);
			pyr_buf[i << 2].height = ((pyr_buf[(i << 2) - (next << 2)].height / 2) | 0);
			pyr_buf[i << 2].getContext("2d").drawImage(pyr_buf[(i << 2) - (next << 2)], 0, 0, pyr_buf[(i << 2) - (next << 2)].width, pyr_buf[(i << 2) - (next << 2)].height, 0, 0, pyr_buf[i << 2].width, pyr_buf[i << 2].height);
			pyr[i << 2] = { "width" : pyr_buf[i << 2].width,
						    "height" : pyr_buf[i << 2].height,
						    "data" : pyr_buf[i << 2].getContext("2d").getImageData(0, 0, pyr_buf[i << 2].width, pyr_buf[i << 2].height).data };
		}
		for (i = (next << 1); i < scale_upto_next_1; i++) {
			pyr_buf[(i << 2) + 1] = document.createElement("canvas");
			pyr_buf[(i << 2) + 1].width = ((pyr_buf[(i << 2) - (next << 2)].width / 2) | 0);
			pyr_buf[(i << 2) + 1].height = ((pyr_buf[(i << 2) - (next << 2)].height / 2) | 0);
			pyr_buf[(i << 2) + 1].getContext("2d").drawImage(pyr_buf[(i << 2) - (next << 2)], 1, 0, pyr_buf[(i << 2) - (next << 2)].width - 1, pyr_buf[(i << 2) - (next << 2)].height, 0, 0, pyr_buf[(i << 2) + 1].width - 2, pyr_buf[(i << 2) + 1].height);
			pyr[(i << 2) + 1] = { "width" : pyr_buf[(i << 2) + 1].width,
								  "height" : pyr_buf[(i << 2) + 1].height,
								  "data" : pyr_buf[(i << 2) + 1].getContext("2d").getImageData(0, 0, pyr_buf[(i << 2) + 1].width, pyr_buf[(i << 2) + 1].height).data };
			pyr_buf[(i << 2) + 2] = document.createElement("canvas");
			pyr_buf[(i << 2) + 2].width = ((pyr_buf[(i << 2) - (next << 2)].width / 2) | 0);
			pyr_buf[(i << 2) + 2].height = ((pyr_buf[(i << 2) - (next << 2)].height / 2) | 0);
			pyr_buf[(i << 2) + 2].getContext("2d").drawImage(pyr_buf[(i << 2) - (next << 2)], 0, 1, pyr_buf[(i << 2) - (next << 2)].width, pyr_buf[(i << 2) - (next << 2)].height - 1, 0, 0, pyr_buf[(i << 2) + 2].width, pyr_buf[(i << 2) + 2].height - 2);
			pyr[(i << 2) + 2] = { "width" : pyr_buf[(i << 2) + 2].width,
								  "height" : pyr_buf[(i << 2) + 2].height,
								  "data" : pyr_buf[(i << 2) + 2].getContext("2d").getImageData(0, 0, pyr_buf[(i << 2) + 2].width, pyr_buf[(i << 2) + 2].height).data };
			pyr_buf[(i << 2) + 3] = document.createElement("canvas");
			pyr_buf[(i << 2) + 3].width = ((pyr_buf[(i << 2) - (next << 2)].width / 2) | 0);
			pyr_buf[(i << 2) + 3].height = ((pyr_buf[(i << 2) - (next << 2)].height / 2) | 0);
			pyr_buf[(i << 2) + 3].getContext("2d").drawImage(pyr_buf[(i << 2) - (next << 2)], 1, 1, pyr_buf[(i << 2) - (next << 2)].width - 1, pyr_buf[(i << 2) - (next << 2)].height - 1, 0, 0, pyr_buf[(i << 2) + 3].width - 2, pyr_buf[(i << 2) + 3].height - 2);
			pyr[(i << 2) + 3] = { "width" : pyr_buf[(i << 2) + 3].width,
								  "height" : pyr_buf[(i << 2) + 3].height,
								  "data" : pyr_buf[(i << 2) + 3].getContext("2d").getImageData(0, 0, pyr_buf[(i << 2) + 3].width, pyr_buf[(i << 2) + 3].height).data };
		}

		var scale_x = 1, scale_y = 1;
		var dx = [0, 1, 0, 1];
		var dy = [0, 0, 1, 1];
		var seq = [];

		var cascade_stage_classifier = cascade.stage_classifier;
		var cascade_stage_classifier_length = cascade_stage_classifier.length;
		var modify_cascade_stage_classifier = new Array(cascade_stage_classifier_length);
		for (i = 0; i < cascade_stage_classifier_length; i++) {
			modify_cascade_stage_classifier[i] = {};
		}
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
				var orig_feature = cascade_stage_classifier_j.feature;
				var modify_cascade_stage_classifier_j = modify_cascade_stage_classifier[j];
				var modify_feature = modify_cascade_stage_classifier_j.feature = new Array(cascade_stage_classifier_j_count);
				for (k = 0; k < cascade_stage_classifier_j_count; k++) {
					var orig_feature_k = orig_feature[k];
					var orig_feature_k_size = orig_feature_k.size;
					var orig_feature_k_px = orig_feature_k.px;
					var orig_feature_k_py = orig_feature_k.py;
					var orig_feature_k_pz = orig_feature_k.pz;
					var orig_feature_k_nx = orig_feature_k.nx;
					var orig_feature_k_ny = orig_feature_k.ny;
					var orig_feature_k_nz = orig_feature_k.nz;
					modify_feature[k] = {"size" : orig_feature_k_size,
										 "px" : new Array(orig_feature_k_size),
										 "pz" : new Array(orig_feature_k_size),
										 "nx" : new Array(orig_feature_k_size),
										 "nz" : new Array(orig_feature_k_size)};
					var modify_feature_k = modify_feature[k];
					for (q = 0; q < orig_feature_k_size; q++) {
						modify_feature_k.px[q] = (orig_feature_k_px[q] << 2) + orig_feature_k_py[q] * step[orig_feature_k_pz[q]];
						modify_feature_k.pz[q] = orig_feature_k_pz[q];
						modify_feature_k.nx[q] = (orig_feature_k_nx[q] << 2) + orig_feature_k_ny[q] * step[orig_feature_k_nz[q]];
						modify_feature_k.nz[q] = orig_feature_k_nz[q];
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
							var modify_cascade_stage_classifier_j = modify_cascade_stage_classifier[j];
							var modify_feature = modify_cascade_stage_classifier_j.feature;
							sum = 0;
							var alpha = cascade_stage_classifier_j.alpha;
							for (k = 0; k < cascade_stage_classifier_j_count; k++) {
								var modify_feature_k = modify_feature[k];
								var modify_feature_k_px = modify_feature_k.px;
								var modify_feature_k_pz = modify_feature_k.pz;
								var modify_feature_k_nx = modify_feature_k.nx;
								var modify_feature_k_nz = modify_feature_k.nz;
								var p, pmin = u8[modify_feature_k_pz[0]][u8o[modify_feature_k_pz[0]] + modify_feature_k_px[0]];
								var n, nmax = u8[modify_feature_k_nz[0]][u8o[modify_feature_k_nz[0]] + modify_feature_k_nx[0]];
								if (pmin <= nmax) {
									sum += alpha[k << 1];
								} else {
									var f, shortcut = true;
									var modify_feature_k_size = modify_feature_k.size;
									for (f = 0; f < modify_feature_k_size; f++) {
										if (modify_feature_k_pz[f] >= 0) {
											p = u8[modify_feature_k_pz[f]][u8o[modify_feature_k_pz[f]] + modify_feature_k_px[f]];
											if (p < pmin) {
												if (p <= nmax) {
													shortcut = false;
													break;
												}
												pmin = p;
											}
										}
										if (modify_feature_k_nz[f] >= 0) {
											n = u8[modify_feature_k_nz[f]][u8o[modify_feature_k_nz[f]] + modify_feature_k_nx[f]];
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
	},

	post : function (params,seq) {
		var i, j;
		var min_neighbors = params.min_neighbors;
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

	}

};
