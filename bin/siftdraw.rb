#!/usr/bin/env ruby

exit unless ARGV.length == 3 or ARGV.length == 4
matches = File.new("/tmp/matches.txt", "w+") if ARGV.length == 4

object_size = Array.new
image_size = Array.new
pairs = Array.new

STDIN.each_line do |line|
	print line
	args = line.split(" ")
	break if args[1] == "keypoints"
	if args[0].include? "x"
		object_size = args[0].split("x")
		object_size = {:width => object_size[0].to_i, :height => object_size[1].to_i}
		image_size = args[2].split("x")
		image_size = {:width => image_size[0].to_i, :height => image_size[1].to_i}
	else
		if matches.nil?
			pairs << {:object => {:x => args[0].to_f, :y => args[1].to_f},
					  :image => {:x => args[3].to_f, :y => args[4].to_f}}
		else
			matches.puts args[0] + " " + args[1] + " " + args[3] + " " + args[4] + "\n"
		end
	end
end

if matches.nil?
	%x[#{sprintf("convert %s -extent %dx%d %s", ARGV[0], object_size[:width] + image_size[:width], [object_size[:height], image_size[:height]].max, ARGV[2])}]
	%x[#{sprintf("composite -gravity southEast %s %s %s", ARGV[1], ARGV[2], ARGV[2])}]
	lines = ""
	pairs.each do |pair|
		lines += sprintf("-draw \"line %d,%d,%d,%d\" ", pair[:object][:x], pair[:object][:y], pair[:image][:x] + object_size[:width], pair[:image][:y])
	end
	%x[convert #{ARGV[2]} -stroke red #{lines}#{ARGV[2]}]
else
	matches.close
	output = %x[#{ARGV[3] + " /tmp/matches.txt"}]
	line = output.split("\n")
	h = Array.new
	h[0] = line[4].split(" ")
	h[0] = [h[0][0].to_f, h[0][1].to_f, h[0][2].to_f]
	h[1] = line[5].split(" ")
	h[1] = [h[1][0].to_f, h[1][1].to_f, h[1][2].to_f]
	h[2] = line[6].split(" ")
	h[2] = [h[2][0].to_f, h[2][1].to_f, h[2][2].to_f]
	points = [{:x => 0, :y => 0},
			  {:x => object_size[:width], :y => 0},
			  {:x => object_size[:width], :y => object_size[:height]},
			  {:x => 0, :y => object_size[:height]}]
	frame = Array.new
	points.each do |point|
		x = h[0][0] * point[:x] + h[0][1] * point[:y] + h[0][2]
		y = h[1][0] * point[:x] + h[1][1] * point[:y] + h[1][2]
		z = h[2][0] * point[:x] + h[2][1] * point[:y] + h[2][2]
		frame << {:x => x / z, :y => y / z}
	end
	%x[#{sprintf("convert %s -stroke red -strokewidth 3 -draw \"line %d,%d,%d,%d\" -draw \"line %d,%d,%d,%d\" -draw \"line %d,%d,%d,%d\" -draw \"line %d,%d,%d,%d\" %s", ARGV[1], frame[0][:x], frame[0][:y], frame[1][:x], frame[1][:y], frame[1][:x], frame[1][:y], frame[2][:x], frame[2][:y], frame[2][:x], frame[2][:y], frame[3][:x], frame[3][:y], frame[3][:x], frame[3][:y], frame[0][:x], frame[0][:y], ARGV[2])}]
end
