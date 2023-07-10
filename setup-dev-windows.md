# How to setup DEV enviroment for Windows

1. VS code
2. Git for Windows
3. Anaconda (pip + virtualenv)

## How to activate Anaconda for Powershell and VS Code terminal

1. add the location of conda to your PATH (if you did not add it via the installer). For me on an "All Users" install this is C:\ProgramData\Anaconda\Scripts
2. from an Administrator Powershell prompt change the Powershell Execution Policy to remote signed i.e. Set-ExecutionPolicy RemoteSigned, or from a non-admin Powershell prompt change the Powershell Execution Policy to remote signed i.e. Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
3. open an Anaconda Prompt and run conda init powershell which will add Conda related startup to a Powershell profile.ps1 somewhere in your user's profile.