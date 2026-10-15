using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Text;
using System.Windows.Forms;

[assembly: AssemblyTitle("LocalGaze")]
[assembly: AssemblyDescription("LocalGaze Eye Tracking Workbench")]
[assembly: AssemblyCompany("LocalGaze")]
[assembly: AssemblyProduct("LocalGaze")]
[assembly: AssemblyVersion("2026.10.15.0")]
[assembly: AssemblyFileVersion("2026.10.15.0")]

internal static class PortableLauncher
{
    private const string BundleDirectory = "__BUNDLE_DIRECTORY__";

    private static string Quote(string value)
    {
        StringBuilder output = new StringBuilder("\"");
        int slashes = 0;
        foreach (char c in value)
        {
            if (c == '\\') { slashes++; continue; }
            if (c == '"') { output.Append('\\', 2 * slashes + 1); output.Append(c); }
            else { output.Append('\\', slashes); output.Append(c); }
            slashes = 0;
        }
        output.Append('\\', 2 * slashes);
        output.Append('"');
        return output.ToString();
    }

    [STAThread]
    private static int Main(string[] args)
    {
        try
        {
            string root = AppDomain.CurrentDomain.BaseDirectory;
            string bundle = Path.GetFullPath(Path.Combine(root, BundleDirectory));
            string executable = Path.Combine(bundle, "LocalGaze.exe");
            if (!File.Exists(executable))
                throw new FileNotFoundException("找不到 LocalGaze 运行环境，请保留以下目录：" + bundle, executable);
            StringBuilder arguments = new StringBuilder();
            foreach (string argument in args)
            {
                if (arguments.Length > 0) arguments.Append(' ');
                arguments.Append(Quote(argument));
            }
            ProcessStartInfo start = new ProcessStartInfo(executable, arguments.ToString());
            start.WorkingDirectory = bundle;
            start.UseShellExecute = false;
            using (Process process = Process.Start(start))
            {
                process.WaitForExit();
                return process.ExitCode;
            }
        }
        catch (Exception error)
        {
            MessageBox.Show(error.Message, "LocalGaze 启动失败", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return 1;
        }
    }
}
