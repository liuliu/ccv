#!/usr/bin/env ruby

require 'pathname'
require 'erb'

tests = []
Dir.glob("**/*.tests.c").each do |fn|
	filename = Pathname.new fn
	objname = String.new fn
	objname[-2, 2] = '.o'
	next if fn == 'all.tests.c'
	tests << {
		:filename => fn,
		:objname => objname,
		:dir => filename.dirname.to_s,
		:prefix => '__' + filename.dirname.to_s.gsub(/[\/\.\\]/, '_') + '_' + filename.basename('.tests.c').to_s.gsub(/\./, '_')
	}
end

def all_mk tests
	all_mk = ERB.new File.read('all.mk.erb')
	all_mk.result binding
end

File.open('all.mk', 'w+') do |f|
	f.write all_mk tests
end
