$ErrorActionPreference = 'Stop'

Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;

public static class InputBridge
{
    [StructLayout(LayoutKind.Sequential)]
    public struct POINT
    {
        public int X;
        public int Y;
    }

    [DllImport("user32.dll")]
    private static extern bool GetCursorPos(out POINT lpPoint);

    [DllImport("user32.dll")]
    private static extern uint MapVirtualKey(uint uCode, uint uMapType);

    [DllImport("user32.dll")]
    private static extern void keybd_event(byte bVk, byte bScan, uint dwFlags, UIntPtr dwExtraInfo);

    [DllImport("user32.dll")]
    private static extern void mouse_event(uint dwFlags, int dx, int dy, int dwData, UIntPtr dwExtraInfo);

    private const uint KEYEVENTF_KEYUP = 0x0002;
    private const uint MOUSEEVENTF_MOVE = 0x0001;
    private const uint MOUSEEVENTF_LEFTDOWN = 0x0002;
    private const uint MOUSEEVENTF_LEFTUP = 0x0004;
    private const uint MOUSEEVENTF_RIGHTDOWN = 0x0008;
    private const uint MOUSEEVENTF_RIGHTUP = 0x0010;
    private const uint MOUSEEVENTF_MIDDLEDOWN = 0x0020;
    private const uint MOUSEEVENTF_MIDDLEUP = 0x0040;
    private const uint MOUSEEVENTF_WHEEL = 0x0800;

    public static int[] GetCursor()
    {
        POINT point;
        if (!GetCursorPos(out point))
        {
            throw new Exception("GetCursorPos failed");
        }

        return new int[] { point.X, point.Y };
    }

    public static void KeyDown(int vk)
    {
        byte scan = (byte)MapVirtualKey((uint)vk, 0);
        keybd_event((byte)vk, scan, 0, UIntPtr.Zero);
    }

    public static void KeyUp(int vk)
    {
        byte scan = (byte)MapVirtualKey((uint)vk, 0);
        keybd_event((byte)vk, scan, KEYEVENTF_KEYUP, UIntPtr.Zero);
    }

    public static void MouseMove(int dx, int dy)
    {
        mouse_event(MOUSEEVENTF_MOVE, dx, dy, 0, UIntPtr.Zero);
    }

    public static void MouseButton(string button, bool down)
    {
        string normalized = (button ?? "left").ToLowerInvariant();
        uint flag;
        if (normalized == "left")
        {
            flag = down ? MOUSEEVENTF_LEFTDOWN : MOUSEEVENTF_LEFTUP;
        }
        else if (normalized == "right")
        {
            flag = down ? MOUSEEVENTF_RIGHTDOWN : MOUSEEVENTF_RIGHTUP;
        }
        else if (normalized == "middle")
        {
            flag = down ? MOUSEEVENTF_MIDDLEDOWN : MOUSEEVENTF_MIDDLEUP;
        }
        else
        {
            throw new Exception("Unsupported mouse button: " + button);
        }

        mouse_event(flag, 0, 0, 0, UIntPtr.Zero);
    }

    public static void Wheel(int amount)
    {
        mouse_event(MOUSEEVENTF_WHEEL, 0, 0, amount, UIntPtr.Zero);
    }
}
"@

function Write-Json($obj) {
    $obj | ConvertTo-Json -Compress -Depth 8 | Write-Output
    [Console]::Out.Flush()
}

Write-Json @{ event = 'ready' }

while (($line = [Console]::In.ReadLine()) -ne $null) {
    if ([string]::IsNullOrWhiteSpace($line)) {
        continue
    }

    $id = $null

    try {
        $msg = $line | ConvertFrom-Json
        $id = $msg.id

        switch ($msg.type) {
            'get_cursor' {
                $point = [InputBridge]::GetCursor()
                Write-Json @{ id = $id; ok = $true; x = $point[0]; y = $point[1] }
            }
            'key_down' {
                [InputBridge]::KeyDown([int]$msg.vk)
                Write-Json @{ id = $id; ok = $true }
            }
            'key_up' {
                [InputBridge]::KeyUp([int]$msg.vk)
                Write-Json @{ id = $id; ok = $true }
            }
            'mouse_move' {
                [InputBridge]::MouseMove([int]$msg.dx, [int]$msg.dy)
                Write-Json @{ id = $id; ok = $true }
            }
            'mouse_down' {
                [InputBridge]::MouseButton([string]$msg.button, $true)
                Write-Json @{ id = $id; ok = $true }
            }
            'mouse_up' {
                [InputBridge]::MouseButton([string]$msg.button, $false)
                Write-Json @{ id = $id; ok = $true }
            }
            'wheel' {
                [InputBridge]::Wheel([int]$msg.amount)
                Write-Json @{ id = $id; ok = $true }
            }
            default {
                throw "Unsupported input command: $($msg.type)"
            }
        }
    }
    catch {
        Write-Json @{ id = $id; ok = $false; error = $_.Exception.Message }
    }
}
