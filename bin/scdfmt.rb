#!/usr/bin/env ruby

exit unless ARGV.length == 0

STDIN.each_line do |line|
	args = line.split(" ")
	if args.count == 2
		print File.dirname(args[0]).to_s + "/" + File.basename(args[0], File.extname(args[0])).to_s + "\n" + args[1] + "\n"
	elsif args.count == 5
		x = args[0].to_i
		y = args[1].to_i
		width = args[2].to_i
		height = args[3].to_i
		confidence = args[4].to_f
		print (width / 2).round.to_s + " " + (height / 2 * 1.6).round.to_s + " 0 " + (x + width / 2).round.to_s + " " + (y + height * 0.3).round.to_s + " " + confidence.to_s + "\n"
	end
end
