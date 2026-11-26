using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.RegularExpressions;
using System.Globalization;

internal static class WindowsScreenshotHelper
{
    private const int SRCCOPY = 0x00CC0020;
    private const int HALFTONE = 4;
    private const int DI_NORMAL = 0x0003;
    private const int CURSOR_SHOWING = 0x00000001;

    private static readonly Regex IdPattern = new Regex("\\\"id\\\"\\s*:\\s*\\\"((?:\\\\.|[^\\\"])*)\\\"", RegexOptions.Compiled);
    private static readonly Regex DisplayPattern = new Regex("\\\"displayId\\\"\\s*:\\s*\\\"((?:\\\\.|[^\\\"])*)\\\"", RegexOptions.Compiled);
    private static readonly Regex IntPattern = new Regex("\\\"(?<name>width|height|quality)\\\"\\s*:\\s*(?<value>-?\\d+)", RegexOptions.Compiled);
    private static MonitorEntry cachedMonitor;
    private static string cachedMonitorName;
    private static ImageCodecInfo cachedJpegCodec;

    [StructLayout(LayoutKind.Sequential)]
    private struct Rect
    {
        public int left;
        public int top;
        public int right;
        public int bottom;
    }

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
    private struct MonitorInfoEx
    {
        public int cbSize;
        public Rect rcMonitor;
        public Rect rcWork;
        public uint dwFlags;

        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 32)]
        public string szDevice;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct Point
    {
        public int x;
        public int y;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct CursorInfo
    {
        public int cbSize;
        public int flags;
        public IntPtr hCursor;
        public Point ptScreenPos;
    }

    private delegate bool MonitorEnumDelegate(IntPtr monitor, IntPtr hdc, IntPtr rect, IntPtr data);

    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    private static extern bool GetMonitorInfo(IntPtr hMonitor, ref MonitorInfoEx info);

    [DllImport("user32.dll")]
    private static extern bool EnumDisplayMonitors(IntPtr hdc, IntPtr clip, MonitorEnumDelegate callback, IntPtr data);

    [DllImport("user32.dll")]
    private static extern IntPtr GetDC(IntPtr window);

    [DllImport("user32.dll")]
    private static extern int ReleaseDC(IntPtr window, IntPtr dc);

    [DllImport("gdi32.dll")]
    private static extern IntPtr CreateCompatibleDC(IntPtr dc);

    [DllImport("gdi32.dll")]
    private static extern IntPtr CreateCompatibleBitmap(IntPtr dc, int width, int height);

    [DllImport("gdi32.dll")]
    private static extern IntPtr SelectObject(IntPtr dc, IntPtr objectHandle);

    [DllImport("gdi32.dll")]
    private static extern bool DeleteDC(IntPtr dc);

    [DllImport("gdi32.dll")]
    private static extern bool DeleteObject(IntPtr objectHandle);

    [DllImport("gdi32.dll")]
    private static extern bool StretchBlt(IntPtr destination, int x, int y, int width, int height, IntPtr source, int sourceX, int sourceY, int sourceWidth, int sourceHeight, int rop);

    [DllImport("gdi32.dll")]
    private static extern int SetStretchBltMode(IntPtr dc, int mode);

    [DllImport("user32.dll")]
    private static extern bool GetCursorInfo(ref CursorInfo info);

    [DllImport("user32.dll")]
    private static extern bool DrawIconEx(IntPtr dc, int x, int y, IntPtr icon, int width, int height, uint step, IntPtr brush, uint flags);

    private sealed class MonitorEntry
    {
        public string Name;
        public Rect Bounds;
    }

    private static string UnescapeJson(string value)
    {
        return value
            .Replace("\\\"", "\"")
            .Replace("\\\\", "\\")
            .Replace("\\/", "/");
    }

    private static string JsonString(string value)
    {
        var builder = new StringBuilder("\"");
        foreach (var character in value ?? "")
        {
            if (character == '\\' || character == '\"')
            {
                builder.Append('\\');
            }
            if (character == '\r')
            {
                builder.Append("\\r");
            }
            else if (character == '\n')
            {
                builder.Append("\\n");
            }
            else
            {
                builder.Append(character);
            }
        }
        builder.Append('"');
        return builder.ToString();
    }

    private static int ReadInt(string request, string name, int fallback)
    {
        foreach (Match match in IntPattern.Matches(request))
        {
            int value;
            if (match.Groups["name"].Value == name && Int32.TryParse(match.Groups["value"].Value, out value))
            {
                return value;
            }
        }
        return fallback;
    }

    private static string ReadId(string request)
    {
        var match = IdPattern.Match(request);
        return match.Success ? UnescapeJson(match.Groups[1].Value) : "";
    }

    private static string ReadDisplayId(string request)
    {
        var match = DisplayPattern.Match(request);
        return match.Success ? UnescapeJson(match.Groups[1].Value) : "";
    }

    private static MonitorEntry FindMonitor(string requestedName)
    {
        if (cachedMonitor != null && String.Equals(cachedMonitorName, requestedName, StringComparison.OrdinalIgnoreCase))
        {
            return cachedMonitor;
        }

        MonitorEntry found = null;
        EnumDisplayMonitors(IntPtr.Zero, IntPtr.Zero, (monitor, hdc, rect, data) =>
        {
            var info = new MonitorInfoEx { cbSize = Marshal.SizeOf(typeof(MonitorInfoEx)) };
            if (GetMonitorInfo(monitor, ref info))
            {
                var entry = new MonitorEntry { Name = info.szDevice, Bounds = info.rcMonitor };
                if (found == null && (String.IsNullOrEmpty(requestedName) || String.Equals(entry.Name, requestedName, StringComparison.OrdinalIgnoreCase)))
                {
                    found = entry;
                }
            }
            return found == null;
        }, IntPtr.Zero);
        if (found != null)
        {
            cachedMonitor = found;
            cachedMonitorName = requestedName;
        }
        return found;
    }

    private static string Capture(string request)
    {
        var totalWatch = Stopwatch.StartNew();
        var id = ReadDisplayId(request);
        var width = Math.Max(1, ReadInt(request, "width", 1280));
        var height = Math.Max(1, ReadInt(request, "height", 720));
        var quality = Math.Max(1, Math.Min(100, ReadInt(request, "quality", 85)));
        var monitor = FindMonitor(id);
        if (monitor == null)
        {
            throw new InvalidOperationException("Display was not found: " + id);
        }
        var monitorMs = totalWatch.Elapsed.TotalMilliseconds;

        var sourceWidth = Math.Max(1, monitor.Bounds.right - monitor.Bounds.left);
        var sourceHeight = Math.Max(1, monitor.Bounds.bottom - monitor.Bounds.top);
        var scale = Math.Min((double)width / sourceWidth, (double)height / sourceHeight);
        var contentWidth = Math.Max(1, (int)Math.Round(sourceWidth * scale));
        var contentHeight = Math.Max(1, (int)Math.Round(sourceHeight * scale));
        var offsetX = (width - contentWidth) / 2;
        var offsetY = (height - contentHeight) / 2;

        var sourceDc = GetDC(IntPtr.Zero);
        if (sourceDc == IntPtr.Zero)
        {
            throw new InvalidOperationException("GetDC failed.");
        }

        IntPtr destinationDc = IntPtr.Zero;
        IntPtr bitmapHandle = IntPtr.Zero;
        IntPtr previousObject = IntPtr.Zero;
        try
        {
            var gdiWatch = Stopwatch.StartNew();
            destinationDc = CreateCompatibleDC(sourceDc);
            bitmapHandle = CreateCompatibleBitmap(sourceDc, width, height);
            if (destinationDc == IntPtr.Zero || bitmapHandle == IntPtr.Zero)
            {
                throw new InvalidOperationException("Could not allocate target bitmap.");
            }

            previousObject = SelectObject(destinationDc, bitmapHandle);
            using (var graphics = Graphics.FromHdc(destinationDc))
            {
                graphics.Clear(Color.Black);
            }
            SetStretchBltMode(destinationDc, HALFTONE);
            if (!StretchBlt(destinationDc, offsetX, offsetY, contentWidth, contentHeight, sourceDc, monitor.Bounds.left, monitor.Bounds.top, sourceWidth, sourceHeight, SRCCOPY))
            {
                throw new InvalidOperationException("StretchBlt failed.");
            }

            var cursorInfo = new CursorInfo { cbSize = Marshal.SizeOf(typeof(CursorInfo)) };
            var cursorIncluded = GetCursorInfo(ref cursorInfo) && (cursorInfo.flags & CURSOR_SHOWING) != 0 && cursorInfo.hCursor != IntPtr.Zero;
            if (cursorIncluded)
            {
                var cursorX = offsetX + (int)Math.Round((cursorInfo.ptScreenPos.x - monitor.Bounds.left) * scale);
                var cursorY = offsetY + (int)Math.Round((cursorInfo.ptScreenPos.y - monitor.Bounds.top) * scale);
                DrawIconEx(destinationDc, cursorX, cursorY, cursorInfo.hCursor, 0, 0, 0, IntPtr.Zero, DI_NORMAL);
            }
            var gdiMs = gdiWatch.Elapsed.TotalMilliseconds;

            var jpegWatch = Stopwatch.StartNew();
            using (var bitmap = Image.FromHbitmap(bitmapHandle))
            using (var stream = new MemoryStream())
            {
                var codec = JpegCodec();
                if (codec == null)
                {
                    throw new InvalidOperationException("JPEG codec was not found.");
                }
                using (var encoderParameters = new EncoderParameters(1))
                {
                    encoderParameters.Param[0] = new EncoderParameter(System.Drawing.Imaging.Encoder.Quality, quality);
                    bitmap.Save(stream, codec, encoderParameters);
                }
                var jpegEncodeMs = jpegWatch.Elapsed.TotalMilliseconds;
                var base64Watch = Stopwatch.StartNew();
                var data = Convert.ToBase64String(stream.ToArray());
                var base64Ms = base64Watch.Elapsed.TotalMilliseconds;
                var totalMs = totalWatch.Elapsed.TotalMilliseconds;
                return "{\"ok\":true,\"mimeType\":\"image/jpeg\",\"width\":" + width + ",\"height\":" + height + ",\"contentLeft\":" + offsetX + ",\"contentTop\":" + offsetY + ",\"contentWidth\":" + contentWidth + ",\"contentHeight\":" + contentHeight + ",\"cursorIncluded\":" + (cursorIncluded ? "true" : "false") + ",\"cursorX\":" + (cursorIncluded ? cursorInfo.ptScreenPos.x : -1) + ",\"cursorY\":" + (cursorIncluded ? cursorInfo.ptScreenPos.y : -1) + ",\"helperTotalMs\":" + JsonNumber(totalMs) + ",\"helperMonitorMs\":" + JsonNumber(monitorMs) + ",\"helperGdiMs\":" + JsonNumber(gdiMs) + ",\"helperJpegEncodeMs\":" + JsonNumber(jpegEncodeMs) + ",\"helperBase64Ms\":" + JsonNumber(base64Ms) + ",\"data\":" + JsonString(data) + "}";
            }
        }
        finally
        {
            if (previousObject != IntPtr.Zero && destinationDc != IntPtr.Zero)
            {
                SelectObject(destinationDc, previousObject);
            }
            if (bitmapHandle != IntPtr.Zero)
            {
                DeleteObject(bitmapHandle);
            }
            if (destinationDc != IntPtr.Zero)
            {
                DeleteDC(destinationDc);
            }
            ReleaseDC(IntPtr.Zero, sourceDc);
        }
    }

    private static ImageCodecInfo JpegCodec()
    {
        if (cachedJpegCodec != null)
        {
            return cachedJpegCodec;
        }
        foreach (var codec in ImageCodecInfo.GetImageEncoders())
        {
            if (codec.MimeType == "image/jpeg")
            {
                cachedJpegCodec = codec;
                return cachedJpegCodec;
            }
        }
        return null;
    }

    private static string JsonNumber(double value)
    {
        return value.ToString("F3", CultureInfo.InvariantCulture);
    }

    private static void WriteError(string id, Exception error)
    {
        Console.WriteLine("{\"ok\":false,\"error\":" + JsonString(error.Message) + "}");
        Console.Out.Flush();
    }

    public static void Main()
    {
        Console.WriteLine("{\"event\":\"ready\",\"backend\":\"windows-gdi-stretch-jpeg\"}");
        Console.Out.Flush();
        string line;
        while ((line = Console.ReadLine()) != null)
        {
            if (String.IsNullOrWhiteSpace(line))
            {
                continue;
            }
            var id = ReadId(line);
            try
            {
                if (line.IndexOf("\"type\":\"capture\"", StringComparison.OrdinalIgnoreCase) < 0)
                {
                    throw new InvalidOperationException("Unsupported screenshot helper command.");
                }
                var response = Capture(line);
                if (!String.IsNullOrEmpty(id))
                {
                    response = response.Insert(1, "\"id\":" + JsonString(id) + ",");
                }
                Console.WriteLine(response);
                Console.Out.Flush();
            }
            catch (Exception error)
            {
                WriteError(id, error);
            }
        }
    }
}
