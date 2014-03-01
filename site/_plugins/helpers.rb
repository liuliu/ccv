require 'uri'

module Liquid
	
	module ExtendedFilters

		def date_to_month(input)
			Date::MONTHNAMES[input]
		end

		def date_to_month_abbr(input)
			Date::ABBR_MONTHNAMES[input]
		end

		def url_utf8_escape(input)
			multi = input.split('/')
			escaped = multi.collect do |x|
				begin
					URI.escape(URI.unescape(x))
				rescue StandardError
					URI.escape(x)
				end
			end
			escaped.join('/')
		end

		def date_to_human_string(input)
			Date::MONTHNAMES[input.month.to_i] + " " + input.day.to_s + case input.day.to_i % 10
				when 1; "st, "
				when 2; "nd, "
				when 3; "rd, "
				else "th, "
			end + input.year.to_s
		end

		def date_to_utc(input)
			input.getutc
		end

	end

	Liquid::Template.register_filter(ExtendedFilters)
end
