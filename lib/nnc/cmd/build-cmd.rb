#!/usr/bin/env ruby

require 'pathname'
require 'erb'
require 'set'
require 'digest'

abort "build-cmd.rb [PATHS]\n" unless ARGV.length > 0

class MacroParser
	attr_reader :parameters, :has_attention

	def initialize name
		@name = name
		@len = @name.length
		@has_attention = false
		@parameters = nil
	end

	def done
		not @has_attention and not @parameters.nil?
	end

	def parse line
		strip_line = line.strip
		if @has_attention
			if @parameters.nil?
				if strip_line.start_with? '('
					@parameters = ''
					strip_line = strip_line[1..-1] 
				elsif strip_line.length == 0
					return nil # Consumed.
				else
					# Doesn't match, reset and skip.
					@has_attention = false
					return line # Continue to parse.
				end
			end
			closing_parenthesis = strip_line.index ')'
			# Cannot find closing parenthesis
			if closing_parenthesis.nil?
				@parameters = @parameters + strip_line
				return nil # Consumed.
			end
			if closing_parenthesis > 0 # Only append if closing parenthesis is not the first character.
				@parameters = @parameters + strip_line[0..closing_parenthesis-1]
			end
			@has_attention = false
			return strip_line[closing_parenthesis+1..-1] # Continue to parse the rest.
		end
		if strip_line.start_with? @name
			rest_line = strip_line[@len..-1].strip
			if rest_line.start_with? '('
				closing_parenthesis = rest_line.index ')'
				unless closing_parenthesis.nil?
					# Find the command, cool.
					@parameters = rest_line[1..closing_parenthesis-1]
					return nil # Denote this line is consumed.
				else
					@parameters = rest_line[1..-1]
					@has_attention = true
					return nil # Denote this line is consumed.
				end
			elsif rest_line.length == 0 # Empty, continue to find the start parenthesis
				@parameters = nil
				@has_attention = true
				return nil # Denote this line is consumed.
			end
		else
			@parameters = nil
			@has_attention = false
		end
		return line
	end
end

class CommandFile
	attr_reader :filename

	def cuda?
		@cuda
	end

	def initialize filename
		@filename = filename
		@cuda = filename.end_with? '.cu' # If the file end with cu, it is a cuda command backend
	end
end

class CommandConfig < CommandFile
	attr_reader :command

	def initialize filename, command
		super filename
		@command = command
	end
end

class CommandBackend < CommandFile
	attr_reader :command, :backend

	def initialize filename, command, backend
		super filename
		@command = command
		@backend = backend
	end
end

$command_files = []
$command_configs = []
$command_backends = []
$command_easy_macros = []

def register_command_backend filename, which_command, which_backend
	print filename + ":" + which_command + ":" + which_backend + "\n"
	abort "Command has to end with either _FORWARD or _BACKWARD\n" unless which_command.end_with? '_FORWARD' or which_command.end_with? '_BACKWARD'
	$command_backends << CommandBackend.new(filename, which_command, which_backend)
end

def register_command_config filename, which_command
	$command_configs << CommandConfig.new(filename, which_command)
end

def register_command_file filename, parameters
	dirname = Pathname.new(filename).dirname.to_s
	parameters = parameters.split(',')
	parameters.each do |fn|
		filename = dirname + '/' + fn.strip
		$command_configs << CommandFile.new(filename)
	end
end

def parse_find_backend filename, which_command, find_backend
	dirname = Pathname.new(filename).dirname.to_s
	find_backend.split(',').each do |fn|
		filename = dirname + '/' + fn.strip
		File.open(filename, 'r') do |f|
			which_backend_parser = MacroParser.new 'REGISTER_COMMAND_BACKEND'
			find_file_parser = MacroParser.new 'FIND_FILE'
			f.each_line do |line|
				find_file_parser.parse line
				if find_file_parser.done
					register_command_file filename, find_file_parser.parameters
				end
				which_backend_parser.parse line
				if which_backend_parser.done
					parameters = which_backend_parser.parameters.split(',')
					abort "Mssing parameters for #{filename} when REGISTER_COMMAND_BACKEND" if parameters.nil? or parameters.length != 2
					register_command_backend filename, which_command, parameters[1].strip if which_command == parameters[0].strip
				end
			end
		end
	end
end

# Find source code in subdirs.
Dir.glob("{#{ARGV.join(',')}}/**/*.{c,cu}").each do |fn|
	File.open(fn, 'r') do |f|
		command_parser = MacroParser.new 'REGISTER_COMMAND'
		find_backend_parser = MacroParser.new 'FIND_BACKEND'
		find_file_parser = MacroParser.new 'FIND_FILE'
		new_macro_parser = MacroParser.new '//@REGISTER_EASY_COMMAND_MACRO'
		parsers = [ command_parser, find_backend_parser, new_macro_parser ]
		which_command = nil
		command_macro = nil
		available_commands = Set.new
		register_easy_command_macros = []
		f.each_line do |line|
			find_file_parser.parse line
			unless command_macro.nil?
				command_macro[:macro] = line
				register_easy_command_macros << command_macro
				command_macro = nil
			end
			parsers.each do |parser|
				break if line.nil?
				line = parser.parse line
			end
			if command_parser.done
				which_command = command_parser.parameters
				available_commands << which_command
			end
			if find_file_parser.done
				register_command_file fn, find_file_parser.parameters
			end
			if find_backend_parser.done
				abort "FIND_BACKEND before REGISTER_COMMAND, illegal, aborting." if which_command.nil?
				register_command_config fn, which_command
				parse_find_backend fn, which_command, find_backend_parser.parameters
			end
			if new_macro_parser.done
				abort "REGISTER_EASY_COMMAND_MACRO contains illegal parameters, aborting." if new_macro_parser.parameters.nil?
				# Grab the next line no matter what.
				command_macro = { :command => new_macro_parser.parameters }
			end
		end
		# Finished looking through the file, emit errors if the macro doesn't have corresponding command.
		register_easy_command_macros.each do |register_easy_command_macro|
			abort "REGSITER_EASY_COMMAND_MACRO refers to command #{register_easy_command_macro[:command]} that is not available in this file." if not available_commands.include? register_easy_command_macro[:command]
			$command_easy_macros << register_easy_command_macro
		end
	end
end

def config_mk command_backends, command_configs, command_files
	cmd_srcs = command_backends.reject(&:cuda?).map(&:filename).uniq + command_configs.reject(&:cuda?).map(&:filename).uniq + command_files.reject(&:cuda?).map(&:filename).uniq
	cuda_cmd_srcs = command_backends.select(&:cuda?).map(&:filename).uniq + command_configs.select(&:cuda?).map(&:filename).uniq + command_files.select(&:cuda?).map(&:filename).uniq
	config_mk = ERB.new File.read('config.mk.erb')
	config_mk.result binding
end

def ccv_nnc_cmd_h command_map
	commands = []
	command_map.each do |command, index|
		commands << "#{command}_FORWARD = 0x#{index.to_s(16)}"
		commands << "#{command}_BACKWARD = 0x#{(index + 1).to_s(16)}"
	end
	ccv_nnc_cmd_h = ERB.new File.read('ccv_nnc_cmd.h.erb')
	ccv_nnc_cmd_h.result binding
end

def ccv_nnc_cmd_easy_h command_easy_macros
	ccv_nnc_cmd_h = ERB.new File.read('ccv_nnc_cmd_easy.h.erb')
	ccv_nnc_cmd_h.result binding
end

def ccv_nnc_backend_h backend_map
	backends = []
	backend_map.each do |backend, index|
		backends << "#{backend} = 0x#{index.to_s(16)}"
	end
	ccv_nnc_backend_h = ERB.new File.read('ccv_nnc_backend.h.erb')
	ccv_nnc_backend_h.result binding
end

def ccv_nnc_cmd_inc command_backends, command_map, backend_map, command_chd, backend_chd
	init_map = []
	command_chd[:map].each do |k, v|
		init_map[v * 2] = {:name => k + '_FORWARD', :cmd => command_map[k].to_s(16)}
		init_map[v * 2 + 1] = {:name => k + '_BACKWARD', :cmd => (command_map[k] + 1).to_s(16)}
	end
	backend_init_map = []
	backend_chd[:map].each do |k, v|
		backend_init_map[v] = {:name => k, :backend => backend_map[k].to_s(16)}
	end
	command_idx = Hash[command_chd[:map].map do |k, v|
		[[k + '_FORWARD', v * 2], [k + '_BACKWARD', v * 2 + 1]]
	end.flatten(1)]
	backend_idx = backend_chd[:map]
	ccv_nnc_cmd_inc = ERB.new File.read('ccv_nnc_cmd.inc.erb')
	ccv_nnc_cmd_inc.result binding
end

File.open('config.mk', 'w+') do |f|
	f.write config_mk($command_backends, $command_configs, $command_files)
end

commands = $command_backends.map(&:command).uniq.map do |command|
	if command.end_with? '_FORWARD'
		command[0..-9]
	elsif command.end_with? '_BACKWARD'
		command[0..-10]
	end
end.uniq

backends = $command_backends.map(&:backend).uniq

command_map = {}
commands.each do |command|
	command_map[command] = (Digest::SHA256.hexdigest(command)[0..7].to_i(16) & ~1)
end

File.open('ccv_nnc_cmd.h', 'w+') do |f|
	f.write ccv_nnc_cmd_h(command_map)
end

File.open('ccv_nnc_cmd_easy.h', 'w+') do |f|
	f.write ccv_nnc_cmd_easy_h($command_easy_macros)
end

# Compute hash parameters and mapping.
# The hash parameter is: the bit offset.
# For Perfect Hashing, I also need to determine the number of bucket for first level hashing.
# This is loosely based on http://cmph.sourceforge.net/papers/esa09.pdf (Hash, displace and compress)
# This is also not a strict follow on the paper, mainly due to the fact that I only have around 32 hash
# functions to choose from, therefore, I need to record the offset for each bucket as well.

# This is a simple hash implementation because we already have a very very good
# hashed 32-bit integer value (sha256). Thus, this implementation basically
# get bit offset idx % n + off such that enables us to have idx * n * off more
# hash functions to select from, though realistically speaking, there are not that
# many independent hash functions out of these (30-ish that I am sure about).
def param_hash val, idx, n, off
	(val >> idx) % n + off
end

def perfect_hashing keys, hashvals
	chd = nil
	for i in 1..keys.count
		for k in 0..30 # For the top most hash function, we have at most 30 to select from.
			bucket = ([nil] * i).map { |_| [] }
			# Assigning bucket.
			keys.each do |key|
				bucket_idx = param_hash hashvals[key], k, i, 0
				bucket[bucket_idx] << key
			end
			# Sorting from the biggest bucket to the smallest bucket.
			bucket_with_index = bucket.each_with_index.map { |v, i| { :idx => i, :keys => v } }
									.reject { |v| v[:keys].count == 0 }
									.sort { |a, b| a[:keys].count <=> b[:keys].count }
									.reverse
			t = [false] * keys.count
			has_solution = true
			bucket_params = [nil] * bucket_with_index.count
			bucket_map = {}
			bucket_with_index.each do |b|
				# Inside each bucket, try to find the hash function that will give perfect result, and the said position is not used.
				# Try parameter sweep now.
				params = nil
				for j in 0..30
					for x in b[:keys].count..keys.count
						vals = b[:keys].map { |key| param_hash hashvals[key], j, x, 0 }
						next unless vals.uniq.count == vals.count
						for y in 0..(keys.count - x)
							avals = vals.map { |v| v + y }
 							# We end up with distinctive values, good.
							# And any of the vals doesn't have t set yet.
							params = { :idx => j, :n => x, :off => y } unless avals.any? { |v| t[v] }
							break unless params.nil?
						end
						break unless params.nil?
					end
					break unless params.nil?
				end
				if params.nil? # Cannot find viable parameters, fail, cannot continue.
					has_solution = false
					break
				end
				b[:keys].each do |key|
					val = param_hash hashvals[key], j, x, y
					bucket_map[key] = val
					t[val] = true
				end
				bucket_params[b[:idx]] = params
			end
			if has_solution
				chd = {
					:top_level_params => { :idx => k, :n => i, :off => 0 },
					:bucket_params => bucket_params,
					:map => bucket_map
				}
				break
			end
		end
		break unless chd.nil?
	end
	chd
end

command_chd = perfect_hashing commands, Hash[command_map.map { |k, v| [k, v >> 1] }]

backend_map = {}
backends.each do |backend|
	backend_map[backend] = Digest::SHA256.hexdigest(backend)[0..7].to_i(16)
end

File.open('ccv_nnc_backend.h', 'w+') do |f|
	f.write ccv_nnc_backend_h(backend_map)
end

backend_chd = perfect_hashing backends, backend_map

File.open('ccv_nnc_cmd.inc', 'w+') do |f|
	f.write ccv_nnc_cmd_inc($command_backends, command_map, backend_map, command_chd, backend_chd)
end
