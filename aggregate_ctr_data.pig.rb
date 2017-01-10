require 'time'
require 'fileutils'

CONFIG_DIR = File.expand_path("../config/ctr_data_generation", __FILE__)
LAST_RUN_FILE = File.join(CONFIG_DIR, "last_run")
INTERVAL = 14 * 24 * 60 * 60 # 2 weeks time interval
NOW = Time.now

class Time
  def hours_to(time)
    diff_in_hours = (time.to_i / 60 / 60) - (self.to_i / 60 / 60)

    (0...diff_in_hours).to_a.map do |hour|
      self + hour * 60 * 60
    end
  end
end

FileUtils.mkdir_p(CONFIG_DIR)

PigRunner.on_success do
  File.open(LAST_RUN_FILE, "w") do |file|
    file.write(NOW)
  end
end

PigRunner.runtime_param :input do

  if File.exists?(LAST_RUN_FILE)
    last_run = Time.parse(File.open(LAST_RUN_FILE).read)
  else
    last_run = NOW - INTERVAL
  end

  dates_to_query = last_run.hours_to(NOW).map do |time|
    time.strftime("%Y/%m/%d/%H")
  end

  case dates_to_query.size
    when 0 then nil
    when 1 then "/history*/ed_reports/hourly/#{dates_to_query.first}/*"
    else "/history*/ed_reports/hourly/{#{dates_to_query.join(",")}}/*"
  end
end
