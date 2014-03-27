#!/usr/bin/env ruby

exit unless ARGV.length == 2

truth = Array.new

File.new(ARGV[0]).each_line do |line|
	truth << line.to_i
end

miss1 = 0
miss5 = 0
i = 0

File.new(ARGV[1]).each_line do |line|
	args = line.split " "
	miss1 += 1 if args[0].to_i != truth[i]
	miss5 += 1 if args[0].to_i != truth[i] and args[2].to_i != truth[i] and args[4].to_i != truth[i] and args[6].to_i != truth[i] and args[8].to_i != truth[i]
	i += 1
end

print ((miss1.to_f / i.to_f * 10000).round / 100.0).to_s + "% (1), " + ((miss5.to_f / i.to_f * 10000).round / 100.0).to_s + "% (5)\n"
