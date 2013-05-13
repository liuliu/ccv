#!/usr/bin/env ruby

exit unless ARGV.length == 2

truth = Hash.new
total = 0

files = Dir.glob(ARGV[0] + '/*.txt')

files.each do |file|
	name = nil;
 	boxes = Array.new
 	File.new(file).each_line do |line|
 		next if line[0] == '#'
 		name = line[line.rindex('/') + 1, line.rindex('"') - (line.rindex('/') + 1)] if line[0, 14].downcase  == "image filename"
 		if line[0, 16].downcase == "bounding box for"
 			i = line.scan(/object\s*(\d+)/)[0][0].to_i
 			coord = line.scan(/\((\d+),\s*(\d+)\)\s*-\s*\((\d+),\s*(\d+)\)/)[0]
 			boxes[i - 1] = { :found => false, :x => coord[0].to_i, :y => coord[1].to_i, :width => coord[2].to_i - coord[0].to_i, :height => coord[3].to_i - coord[1].to_i }
 		end
 	end
 	truth[name] = boxes;
 	total += boxes.length;
end

fa = 0
tp = 0

File.new(ARGV[1]).each_line do |line|
	args = line.split " "
	name = args[0].to_s
	name = args[0][args[0].rindex('/') + 1, args[0].length - (args[0].rindex('/') + 1)]
	if !truth[name]
		fa += 1
	else
		x = args[1].to_i
		y = args[2].to_i
		width = args[3].to_i
		height = args[4].to_i
		outlier = -1
		truth[name].each do |obj|
			opx_min = [obj[:x], x].max
			opy_min = [obj[:y], y].max
			opx_max = [obj[:x] + obj[:width], x + width].min
			opy_max = [obj[:y] + obj[:height], y + height].min
			r0 = [opx_max - opx_min, 0].max * [opy_max - opy_min, 0].max
			r1 = [obj[:width] * obj[:height], width * height].max * 0.5
			if r0 > r1
				outlier = obj[:found] ? 0 : 1
				obj[:found] = true
				break
			end
		end
		case outlier
		when -1 then fa += 1
		when 1 then tp += 1
		end
	end
end

print ((tp.to_f / total.to_f * 10000).round / 100.0).to_s + "% ("+ fa.to_s + ")\n"
