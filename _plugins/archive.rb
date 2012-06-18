module Jekyll

	class ArchiveIndex < Page
		def initialize(site, base, dir, type)
			@site = site
			@base = base
			@dir = dir
			@name = 'index.html'

			self.process(@name)
			self.read_yaml(File.join(base, '_layouts'), type + '.html')

			year, month, day = dir.split('/')
			self.data['year'] = year.to_i
			self.data['month'] = month.to_i if month
			self.data['day'] = day.to_i if day
		end

		def collect(collated_posts, older, newer)
			self.data['collated_posts'] = collated_posts
			self.data['previous'] = older
			self.data['next'] = newer
		end

	end

	class ArchiveGenerator < Generator
		safe true
		attr_accessor :collated_posts
		attr_accessor :lbyear
		attr_accessor :ubyear

		def generate(site)
			self.collated_posts = collate(site)

			self.collated_posts.keys.each do |y|
				if site.layouts.key? 'archive_yearly'
					previous_yearly = nil
					y.downto(self.lbyear) do |py|
						if self.collated_posts.key? py
							previous_yearly = Date.civil(py.to_i)
							break
						end
					end
					next_yearly = nil
					y.upto(self.ubyear) do |ny|
						if self.collated_posts.key? ny
							next_yearly = Date.civil(ny.to_i)
							break
						end
					end
					write_archive_index(site, y.to_s, 'archive_yearly', self.collated_posts, previous_yearly, next_yearly)
				end
				self.collated_posts[ y ].keys.each do |m|
	 				if site.layouts.key? 'archive_monthly'
						previous_monthly = nil
						py, pm = y, m
						while py >= self.lbyear do
							pm = pm - 1
							py, pm = py - 1, 12 if pm < 1
							if self.collated_posts.key? py and self.collated_posts[ py ].key? pm
								previous_monthly = Date.civil(py.to_i, pm.to_i)
								break
							end
						end
						next_monthly = nil
						ny, nm = y, m
						while ny <= self.ubyear do
							nm = nm + 1
							ny, nm = py + 1, 1 if nm > 12
							if self.collated_posts.key? ny and self.collated_posts[ ny ].key? nm
								next_monthly = Date.civil(ny.to_i, nm.to_i)
								break
							end
						end
						write_archive_index(site, "%04d/%02d" % [ y.to_s, m.to_s ], 'archive_monthly', self.collated_posts, previous_monthly, next_monthly)
					 end
					if site.layouts.key? 'archive_daily'
						self.collated_posts[ y ][ m ].keys.each do |d|
							previous_daily = nil
							py, pm, pd = y, m, d
							while py >= self.lbyear do
								pd = pd - 1
								pm, pd = pm - 1, 31 if pd < 1
								py, pm = py - 1, 12 if pm < 1
								if self.collated_posts.key? py and self.collated_posts[ py ].key? pm and self.collated_posts[ py ][ pm ].size > 0
									previous_daily = Date.civil(py.to_i, pm.to_i, pd.to_i)
									break
								end
							end
							next_daily = nil
							ny, nm, nd = y, m, d
							while ny <= self.ubyear do
								nd = nd + 1
								nm, nd = pm + 1, 1 if pd > 31
								ny, nm = py + 1, 1 if pm > 12
								if self.collated_posts.key? ny and self.collated_posts[ ny ].key? nm and self.collated_posts[ ny ][ nm ].size > 0
									next_daily = Date.civil(ny.to_i, nm.to_i, nd.to_i)
									break
								end
							end
							write_archive_index(site, "%04d/%02d/%02d" % [ y.to_s, m.to_s, d.to_s ], 'archive_daily', self.collated_posts, previous_daily, next_daily)
						end
					end
				end
			end
		end

		def write_archive_index(site, dir, type, collated_posts, newer, older)
			archive = ArchiveIndex.new(site, site.source, dir, type)
			archive.collect(collated_posts, newer, older)
			archive.render(site.layouts, site.site_payload)
			archive.write(site.dest)
			site.static_files << archive
		end

		def collate(site)
			collated_posts = {}
			self.ubyear = self.lbyear = nil
			site.posts.reverse.each do |post|
				y, m, d = post.date.year, post.date.month, post.date.day
				self.lbyear = y if self.lbyear == nil or y < self.lbyear
				self.ubyear = y if self.ubyear == nil or y > self.ubyear
				collated_posts[ y ] = {} unless collated_posts.key? y
				collated_posts[ y ][ m ] = {} unless collated_posts[y].key? m
				collated_posts[ y ][ m ][ d ] = [] unless collated_posts[ y ][ m ].key? d
				collated_posts[ y ][ m ][ d ].push(post) unless collated_posts[ y ][ m ][ d ].include?(post)
			end
			return collated_posts
		end
	end
end
