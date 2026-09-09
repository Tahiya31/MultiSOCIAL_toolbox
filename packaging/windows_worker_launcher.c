/*
 * Native trampoline for the private Windows analysis worker.
 *
 * The packaged GUI never shares native runtime files with this CPython worker.
 * For GUI-originated launches the worker speaks on an authenticated loopback
 * socket, so this launcher can create it without inheriting GUI standard
 * handles. Direct command-line diagnostics retain their stdio protocol.
 */

#define _WIN32_WINNT 0x0A00
#define UNICODE
#define _UNICODE
#include <windows.h>
#include <strsafe.h>
#include <wchar.h>

#define LAUNCHER_NAME L"MultiSOCIAL-Worker-Launcher.exe"
#define WORKER_PYTHON L"python.exe"
#define WORKER_SCRIPT L"app\\analysis_worker.py"
#define WORKER_RUNTIME_ROOT_ENV L"MULTISOCIAL_WORKER_RUNTIME_ROOT"

static void reset_pyinstaller_environment(void) {
    const WCHAR *const names[] = {
        L"PYINSTALLER_RESET_ENVIRONMENT",
        L"_PYI_APPLICATION_HOME_DIR",
        L"_PYI_ARCHIVE_FILE",
        L"_PYI_PARENT_PROCESS_LEVEL",
        L"_PYI_SPLASH_IPC",
        L"_MEIPASS2",
    };
    size_t index;

    /* The GUI parent is a separate PyInstaller application.  This static
       launcher must not pass its private bootloader state to the worker. */
    for (index = 0; index < ARRAYSIZE(names); ++index) {
        SetEnvironmentVariableW(names[index], NULL);
    }
}

static int fail(DWORD error) {
    char message[128];
    HANDLE standard_error;
    DWORD written;

    /* stderr is a byte stream.  Keep this ASCII so Python and CI can report it. */
    StringCchPrintfA(message, ARRAYSIZE(message), "Worker launcher failed (%lu)\n", error);
    standard_error = GetStdHandle(STD_ERROR_HANDLE);
    if (standard_error != INVALID_HANDLE_VALUE && standard_error != NULL) {
        WriteFile(
            standard_error,
            message,
            (DWORD)lstrlenA(message),
            &written,
            NULL
        );
    }
    return 1;
}

static int wait_for_child(PROCESS_INFORMATION *process_info) {
    DWORD exit_code = 1;

    CloseHandle(process_info->hThread);
    WaitForSingleObject(process_info->hProcess, INFINITE);
    if (!GetExitCodeProcess(process_info->hProcess, &exit_code)) {
        exit_code = 1;
    }
    CloseHandle(process_info->hProcess);
    return (int)exit_code;
}

static BOOL contains_non_ascii(const WCHAR *value) {
    while (*value != L'\0') {
        if (*value > 0x7f) {
            return TRUE;
        }
        ++value;
    }
    return FALSE;
}

static BOOL get_ascii_short_path(const WCHAR *source, WCHAR *destination, DWORD size) {
    DWORD length = GetShortPathNameW(source, destination, size);
    return length > 0 && length < size && !contains_non_ascii(destination);
}

static BOOL create_ascii_drive_alias(
    const WCHAR *source,
    WCHAR *alias,
    DWORD alias_size,
    WCHAR *target,
    DWORD target_size
) {
    DWORD drives = GetLogicalDrives();
    DWORD last_error = ERROR_BUSY;
    WCHAR device_name[3] = {L'\0', L':', L'\0'};
    int index;

    /* DefineDosDevice's raw target is an NT path: \??\C:\..., not \\??\C:\.... */
    if (drives == 0 || StringCchPrintfW(target, target_size, L"\\??\\%s", source) != S_OK) {
        return FALSE;
    }
    for (index = 25; index >= 3; --index) {
        if ((drives & (1u << index)) != 0) {
            continue;
        }
        device_name[0] = (WCHAR)(L'A' + index);
        if (DefineDosDeviceW(
                DDD_RAW_TARGET_PATH | DDD_NO_BROADCAST_SYSTEM,
                device_name,
                target
            )) {
            return StringCchCopyW(alias, alias_size, device_name) == S_OK;
        }
        last_error = GetLastError();
    }
    SetLastError(last_error);
    return FALSE;
}

static void remove_ascii_drive_alias(const WCHAR *alias, const WCHAR *target) {
    DefineDosDeviceW(
        DDD_REMOVE_DEFINITION | DDD_EXACT_MATCH_ON_REMOVE | DDD_NO_BROADCAST_SYSTEM,
        alias,
        target
    );
}

int wmain(void) {
    WCHAR launcher_path[MAX_PATH];
    WCHAR worker_python[MAX_PATH];
    WCHAR worker_script[MAX_PATH];
    WCHAR child_directory[MAX_PATH];
    WCHAR runtime_root[MAX_PATH];
    WCHAR drive_alias[3] = {L'\0', L'\0', L'\0'};
    WCHAR drive_target[MAX_PATH + 5];
    WCHAR command_line[MAX_PATH * 2 + 64];
    WCHAR *separator;
    STARTUPINFOW startup_info;
    PROCESS_INFORMATION process_info;
    BOOL socket_protocol;
    BOOL drive_alias_created = FALSE;
    DWORD error;
    DWORD launcher_length;
    int exit_code;

    launcher_length = GetModuleFileNameW(NULL, launcher_path, ARRAYSIZE(launcher_path));
    if (launcher_length == 0) {
        return fail(GetLastError());
    }
    if (launcher_length >= ARRAYSIZE(launcher_path)) {
        return fail(ERROR_INSUFFICIENT_BUFFER);
    }
    separator = wcsrchr(launcher_path, L'\\');
    if (separator == NULL) {
        return fail(ERROR_BAD_PATHNAME);
    }
    *separator = L'\0';
    if (contains_non_ascii(launcher_path)) {
        if (!get_ascii_short_path(launcher_path, runtime_root, ARRAYSIZE(runtime_root))) {
            if (!create_ascii_drive_alias(
                    launcher_path,
                    drive_alias,
                    ARRAYSIZE(drive_alias),
                    drive_target,
                    ARRAYSIZE(drive_target)
                )) {
                return fail(GetLastError());
            }
            if (StringCchCopyW(runtime_root, ARRAYSIZE(runtime_root), drive_alias) != S_OK) {
                remove_ascii_drive_alias(drive_alias, drive_target);
                return fail(ERROR_BUFFER_OVERFLOW);
            }
            drive_alias_created = TRUE;
        }
    } else if (StringCchCopyW(runtime_root, ARRAYSIZE(runtime_root), launcher_path) != S_OK) {
        return fail(ERROR_BUFFER_OVERFLOW);
    }
    if (StringCchPrintfW(worker_python, ARRAYSIZE(worker_python), L"%s\\%s", runtime_root, WORKER_PYTHON) != S_OK) {
        if (drive_alias_created) {
            remove_ascii_drive_alias(drive_alias, drive_target);
        }
        return fail(ERROR_BUFFER_OVERFLOW);
    }
    if (StringCchPrintfW(worker_script, ARRAYSIZE(worker_script), L"%s\\%s", runtime_root, WORKER_SCRIPT) != S_OK) {
        if (drive_alias_created) {
            remove_ascii_drive_alias(drive_alias, drive_target);
        }
        return fail(ERROR_BUFFER_OVERFLOW);
    }
    if (GetFileAttributesW(worker_python) == INVALID_FILE_ATTRIBUTES || GetFileAttributesW(worker_script) == INVALID_FILE_ATTRIBUTES) {
        error = GetLastError();
        if (drive_alias_created) {
            remove_ascii_drive_alias(drive_alias, drive_target);
        }
        return fail(error);
    }

    socket_protocol = GetEnvironmentVariableW(
        L"MULTISOCIAL_WORKER_PROTOCOL_HOST", NULL, 0
    ) > 0;
    if (!SetEnvironmentVariableW(WORKER_RUNTIME_ROOT_ENV, runtime_root)) {
        error = GetLastError();
        if (drive_alias_created) {
            remove_ascii_drive_alias(drive_alias, drive_target);
        }
        return fail(error);
    }
    /* Reset both legacy and modern process DLL-directory mechanisms. */
    SetDllDirectoryW(NULL);
    SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    reset_pyinstaller_environment();
    /* OpenSMILE's Windows bridge incorrectly encodes its working directory as
       ASCII. The installed application path may legitimately contain Unicode.
       Socket-mode workers use absolute interpreter, module, asset, and output
       paths, so a stable system directory removes that native-library limit
       without changing any user-visible input or output path. */
    if (socket_protocol) {
        DWORD directory_length = GetWindowsDirectoryW(
            child_directory, ARRAYSIZE(child_directory)
        );
        if (directory_length == 0 || directory_length >= ARRAYSIZE(child_directory)) {
            error = directory_length == 0 ? GetLastError() : ERROR_INSUFFICIENT_BUFFER;
            if (drive_alias_created) {
                remove_ascii_drive_alias(drive_alias, drive_target);
            }
            return fail(error);
        }
    } else if (StringCchCopyW(child_directory, ARRAYSIZE(child_directory), runtime_root) != S_OK) {
        if (drive_alias_created) {
            remove_ascii_drive_alias(drive_alias, drive_target);
        }
        return fail(ERROR_BUFFER_OVERFLOW);
    }

    ZeroMemory(&startup_info, sizeof(startup_info));
    startup_info.cb = sizeof(startup_info);
    if (!socket_protocol) {
        startup_info.dwFlags = STARTF_USESTDHANDLES;
        startup_info.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
        startup_info.hStdOutput = GetStdHandle(STD_OUTPUT_HANDLE);
        startup_info.hStdError = GetStdHandle(STD_ERROR_HANDLE);
    }
    ZeroMemory(&process_info, sizeof(process_info));
    if (StringCchPrintfW(command_line, ARRAYSIZE(command_line), L"\"%s\" -I \"%s\"", worker_python, worker_script) != S_OK) {
        if (drive_alias_created) {
            remove_ascii_drive_alias(drive_alias, drive_target);
        }
        return fail(ERROR_BUFFER_OVERFLOW);
    }
    if (!CreateProcessW(
            worker_python,
            command_line,
            NULL,
            NULL,
            socket_protocol ? FALSE : TRUE,
            CREATE_NO_WINDOW,
            NULL,
            child_directory,
            &startup_info,
            &process_info)) {
        error = GetLastError();
        if (drive_alias_created) {
            remove_ascii_drive_alias(drive_alias, drive_target);
        }
        return fail(error);
    }
    exit_code = wait_for_child(&process_info);
    if (drive_alias_created) {
        remove_ascii_drive_alias(drive_alias, drive_target);
    }
    return exit_code;
}
