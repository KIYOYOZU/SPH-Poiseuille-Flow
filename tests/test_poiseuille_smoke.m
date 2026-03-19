project_dir = fileparts(fileparts(mfilename('fullpath')));
source_config = fullfile(project_dir, 'config.ini');

run_tag = ['smoke_', datestr(now, 'yyyymmdd_HHMMSSFFF')];
temp_dir = fullfile(project_dir, 'build', run_tag);
if ~exist(temp_dir, 'dir')
    mkdir(temp_dir);
end

config_text = fileread(source_config);
config_text = regexprep(config_text, 'end_time\s*=\s*[^\r\n]+', 'end_time = 0.05');
config_text = regexprep(config_text, 'output_interval\s*=\s*[^\r\n]+', 'output_interval = 0.05');
config_text = regexprep(config_text, 'sort_interval\s*=\s*[^\r\n]+', 'sort_interval = 20');
config_text = regexprep(config_text, 'restart_from_file\s*=\s*[^\r\n]+', 'restart_from_file = 0');

test_config = fullfile(temp_dir, 'config.ini');
fid = fopen(test_config, 'w');
assert(fid >= 0, '无法创建测试配置文件: %s', test_config);
cleanup_fid = onCleanup(@() fclose(fid));
fwrite(fid, config_text, 'char');
clear cleanup_fid;

setenv('SPH_CONFIG_OVERRIDE', test_config);
setenv('SPH_RESTART_PATH_OVERRIDE', fullfile(temp_dir, 'restart.mat'));
setenv('SPH_RESULT_PNG_OVERRIDE', fullfile(temp_dir, 'SPH_Poiseuille_result.png'));
setenv('SPH_PROFILE_PNG_OVERRIDE', fullfile(temp_dir, 'SPH_centerline_profile_evolution.png'));

run(fullfile(project_dir, 'SPH_Poiseuille.m'));

restart_override = getenv('SPH_RESTART_PATH_OVERRIDE');
result_png_override = getenv('SPH_RESULT_PNG_OVERRIDE');
profile_png_override = getenv('SPH_PROFILE_PNG_OVERRIDE');

assert(exist(restart_override, 'file') == 2, '未生成 restart 覆盖输出。');
assert(exist(result_png_override, 'file') == 2, '未生成结果图覆盖输出。');
assert(exist(profile_png_override, 'file') == 2, '未生成剖面演化图覆盖输出。');

restart_data = load(restart_override, 'state');
assert(isfield(restart_data, 'state'), 'restart 文件缺少 state。');
assert(isfield(restart_data.state, 't') && restart_data.state.t > 0.0, 'restart 状态时间无效。');

disp('test_poiseuille_smoke: PASS');
