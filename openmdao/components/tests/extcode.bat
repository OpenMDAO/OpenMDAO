@echo off
rem usage: extcode.bat output_filename
rem
rem Just write 'test data' to the specified output file

set DATA=test data
set OUT_FILE=%1

echo %DATA%>>%OUT_FILE%
