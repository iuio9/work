package com.qczy.model.entity.domain;

import cn.hutool.core.util.NumberUtil;
import com.qczy.utils.FileFormatSizeUtils;
import oshi.SystemInfo;
import oshi.hardware.CentralProcessor;
import oshi.hardware.CentralProcessor.TickType;
import oshi.hardware.GlobalMemory;
import oshi.hardware.GraphicsCard;
import oshi.hardware.HardwareAbstractionLayer;
import oshi.software.os.FileSystem;
import oshi.software.os.OSFileStore;
import oshi.software.os.OperatingSystem;
import oshi.util.Util;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Properties;

/**
 * 服务器相关信息
 * 
 * @author ruoyi
 */
public class Server
{
    private static final int OSHI_WAIT_SECOND = 1000;


    /**
     * GPU相关信息
     */
    private Gpu gpu = new Gpu();
    /**
     * CPU相关信息
     */
    private Cpu cpu = new Cpu();

    /**
     * 內存相关信息
     */
    private Mem mem = new Mem();

    /**
     * 服务器相关信息
     */
    private Sys sys = new Sys();
    /**
     * 磁盘汇总
     */
    private SysFileInfo sysFileInfo = new SysFileInfo();


    /**
     * 磁盘相关信息
     */
//    private List<SysFile> sysFiles = new LinkedList<SysFile>();

    public Cpu getCpu()
    {
        return cpu;
    }

    public Gpu getGpu() {
        return gpu;
    }

    public void setGpu(Gpu gpu) {
        this.gpu = gpu;
    }

    public void setCpu(Cpu cpu)
    {
        this.cpu = cpu;
    }

    public Mem getMem()
    {
        return mem;
    }

    public void setMem(Mem mem)
    {
        this.mem = mem;
    }

    public Sys getSys()
    {
        return sys;
    }

    public void setSys(Sys sys)
    {
        this.sys = sys;
    }

//    public List<SysFile> getSysFiles()
//    {
//        return sysFiles;
//    }
//
//    public void setSysFiles(List<SysFile> sysFiles)
//    {
//        this.sysFiles = sysFiles;
//    }

    public static boolean isNvidiaSmiAvailable() {
        try {
            // 执行 nvidia-smi 命令
            Process process = Runtime.getRuntime().exec("nvidia-smi");

            // 读取命令输出，检测是否存在
            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            String line;
            StringBuilder output = new StringBuilder();

            while ((line = reader.readLine()) != null) {
                output.append(line).append("\n");
            }

            // 等待命令执行完成
            int exitCode = process.waitFor();
            if (exitCode == 0 && output.toString().contains("NVIDIA-SMI")) {
                return true; // 如果成功执行并且输出包含 "NVIDIA-SMI"，则表示命令存在
            }
        } catch (Exception e) {
            // 异常处理：命令不存在或执行失败
            System.err.println("Error while checking for nvidia-smi: " + e.getMessage());
        }

        return false; // 如果 nvidia-smi 不存在或执行失败
    }
    public void copyTo() throws Exception
    {
        SystemInfo si = new SystemInfo();
        HardwareAbstractionLayer hal = si.getHardware();
        OperatingSystem os = si.getOperatingSystem();
        // 获取 GPU 信息
        List<GraphicsCard> graphicsCards = hal.getGraphicsCards();

        if (isNvidiaSmiAvailable()) {
            System.out.println("系统包含 GPU。");
            setGpuInfo(graphicsCards);
        }
        setCpuInfo(hal.getProcessor());

        setMemInfo(hal.getMemory());

        setSysInfo();

//        setJvmInfo();

        setSysFiles(si.getOperatingSystem());
    }


    public static boolean isUbuntu(OperatingSystem os) {
        // 检查操作系统的名称是否包含 "Ubuntu"
        return os.getFamily().equalsIgnoreCase("Linux");
    }

    public static boolean hasGPU(SystemInfo si) {
        // 使用 OSHI 获取 GPU 信息，如果列表不为空则表示存在 GPU
        List<GraphicsCard> gpus = si.getHardware().getGraphicsCards();
        return !gpus.isEmpty();
    }

    public static String getGpuUsage() {
        StringBuilder result = new StringBuilder();
        try {
            // 执行 nvidia-smi 命令来查询 GPU 使用率和显存使用情况

            Process process = Runtime.getRuntime().exec("nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits");
            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            String line;
            while ((line = reader.readLine()) != null) {
                result.append(line).append("\n");
            }
            process.waitFor();
        } catch (Exception e) {
            e.printStackTrace();
        }
        return result.toString();
    }

    public static void main(String[] args) {
        String s = "0, 7642, 81920\n" +
                "0, 3, 81920\n" +
                "0, 3, 81920\n" +
                "0, 3, 81920\n";
        String[] split = s.split("\n");
//        BigInteger l = NumberUtil.newBigInteger("2.561765376E10");
        BigDecimal bigDecimal = NumberUtil.toBigDecimal("2.561765376E10");
        System.out.println(bigDecimal);
        System.out.println(bigDecimal.longValue());
        BigInteger b = new BigInteger(""+bigDecimal.longValue());
        System.out.println("======="+b);

        long l = bigDecimal.longValueExact();

        System.out.println(l);

        double v = Double.parseDouble("2.561765376E10");
        BigDecimal bg=new BigDecimal(v+"");
        System.out.println(bg);
//        System.out.println(l);

        System.out.println(Arrays.toString(s.split("/n")));

    }

    private void setGpuInfo(List<GraphicsCard> graphicsCards) {
       try{
           List<GPUInfo> gpuInfoList = new ArrayList<>();
           //GPU 利用率：显示 GPU 的当前利用率。
           //已用显存：显示当前已用的显存量。
           //总显存：显示 GPU 的总显存量。
           String gpuUsage = getGpuUsage();
           System.out.println("============================"+gpuUsage);
           if(gpuUsage.equalsIgnoreCase("")){
               return;
           }
           String[] gpuUsageArr = gpuUsage.split("\n");

           System.out.println("============================"+ Arrays.toString(gpuUsageArr));

           long usedVRamTotal = 0;
           double coreUsageTotal =0.0;
           double totalUsage =0.0;

           // 模拟获取 GPU 使用率和已用显存（这里是静态值，可替换为动态获取）
           int i = 0;
           // 封装 GPU 信息到 GPUInfo 类

           for (String s : gpuUsageArr) {
               String[] split = s.split(",");
               long usedVRam = Long.parseLong(split[1].replaceAll(" ", ""));  // 假设使用了 1GB 的显存
               long usage = Long.parseLong(split[0].replaceAll(" ", ""));  // 假设使用了 1GB 的显存
               double coreUsage = Double.parseDouble(split[2].replaceAll(" ", ""));  // 假设核心使用率为 75.5%
               usedVRamTotal += usedVRam;
               coreUsageTotal+=coreUsage;
               totalUsage+= usage;
           }
//           for (GraphicsCard gc : graphicsCards) {
//               System.out.println("graphicsCards======"+graphicsCards.size());
//               System.out.println("iiiiiii======"+i);
//               long usedVRam =0;
//               double coreUsage=0;
//               if(i < graphicsCards.size()-1){
//                   usedVRam = Long.parseLong(gpuUsageArr[i].split(",")[1].replaceAll(" ", ""));  // 假设使用了 1GB 的显存
//                   coreUsage = Double.parseDouble(gpuUsageArr[i].split(",")[2].replaceAll(" ", ""));  // 假设核心使用率为 75.5%
//                   usedVRamTotal += usedVRam;
//                   coreUsageTotal+=coreUsage;
//               }
//              i++;
//               GPUInfo gpuInfo = new GPUInfo(
//                       gc.getName(),
//                       gc.getVendor(),
//                       gc.getDeviceId(),
//                       gc.getVersionInfo(),
//                       gc.getVRam(),
//                       usedVRam,  // 实际项目中应动态获取
//                       coreUsage  // 实际项目中应动态获取
//               );
//               gpuInfoList.add(gpuInfo);
//           }
           System.out.println("gpu====================usedVRamTotal"+usedVRamTotal);
           System.out.println("gpu====================coreUsageTotal"+coreUsageTotal);
           System.out.println("gpu====================usedVRamTotal"+(coreUsageTotal - usedVRamTotal));
           gpu.setTotal(coreUsageTotal);
//        gpu.setSys(cSys);
           gpu.setUsed(usedVRamTotal);
//        gpu.setWait(iowait);
           gpu.setFree(coreUsageTotal - usedVRamTotal);
           gpu.setUsage(NumberUtil.div(totalUsage,gpuUsageArr.length));
       }catch (Exception e){
           e.printStackTrace();
       }
    }

    /**
     * 设置CPU信息
     */
    private void setCpuInfo(CentralProcessor processor)
    {
        // CPU信息
        long[] prevTicks = processor.getSystemCpuLoadTicks();
        Util.sleep(OSHI_WAIT_SECOND);
        long[] ticks = processor.getSystemCpuLoadTicks();
        long nice = ticks[TickType.NICE.getIndex()] - prevTicks[TickType.NICE.getIndex()];
        long irq = ticks[TickType.IRQ.getIndex()] - prevTicks[TickType.IRQ.getIndex()];
        long softirq = ticks[TickType.SOFTIRQ.getIndex()] - prevTicks[TickType.SOFTIRQ.getIndex()];
        long steal = ticks[TickType.STEAL.getIndex()] - prevTicks[TickType.STEAL.getIndex()];
        long cSys = ticks[TickType.SYSTEM.getIndex()] - prevTicks[TickType.SYSTEM.getIndex()];
        long user = ticks[TickType.USER.getIndex()] - prevTicks[TickType.USER.getIndex()];
        long iowait = ticks[TickType.IOWAIT.getIndex()] - prevTicks[TickType.IOWAIT.getIndex()];
        long idle = ticks[TickType.IDLE.getIndex()] - prevTicks[TickType.IDLE.getIndex()];
        long totalCpu = user + nice + cSys + idle + iowait + irq + softirq + steal;
//        cpu.setCpuNum(processor.getLogicalProcessorCount());
        cpu.setTotal(totalCpu);
//        cpu.setSys(cSys);
        cpu.setUsed(totalCpu - idle);
        cpu.setFree(idle);
    }

    /**
     * 设置内存信息
     */
    private void setMemInfo(GlobalMemory memory)
    {
        mem.setTotal(memory.getTotal());
        mem.setUsed(memory.getTotal() - memory.getAvailable());
        mem.setFree(memory.getAvailable());
    }

    /**
     * 设置服务器信息
     */
    private void setSysInfo()
    {
        Properties props = System.getProperties();
        sys.setComputerName(IpUtils.getHostName());
        sys.setComputerIp(IpUtils.getHostIp());
        sys.setOsName(props.getProperty("os.name"));
        sys.setOsArch(props.getProperty("os.arch"));
        sys.setUserDir(props.getProperty("user.dir"));
    }

    /**
     * 设置磁盘信息
     */
    private void setSysFiles(OperatingSystem os)
    {
        FileSystem fileSystem = os.getFileSystem();
        List<OSFileStore> fsArray = fileSystem.getFileStores();
        long freeTotal = 0;
        long disktotal=0;
        long usedtotal=0;
        for (OSFileStore fs : fsArray)
        {
            long free = fs.getUsableSpace();
            long total = fs.getTotalSpace();
            long used = total - free;
            SysFile sysFile = new SysFile();
            sysFile.setDirName(fs.getMount());
            sysFile.setSysTypeName(fs.getType());
            sysFile.setTypeName(fs.getName());
            sysFile.setTotal(convertFileSize(total));
            sysFile.setFree(convertFileSize(free));
            sysFile.setUsed(convertFileSize(used));
            sysFile.setUsage(Arith.mul(Arith.div(used, total, 4), 100));
//            sysFiles.add(sysFile);
            disktotal+= total;
            usedtotal+=used;
            freeTotal+=free;
        }
        sysFileInfo.setTotal(disktotal);
        sysFileInfo.setFree(freeTotal);
        sysFileInfo.setUsed(usedtotal);
    }

    public SysFileInfo getSysFileInfo() {
        return sysFileInfo;
    }

    public void setSysFileInfo(SysFileInfo sysFileInfo) {
        this.sysFileInfo = sysFileInfo;
    }

    /**
     * 字节转换
     * 
     * @param size 字节大小
     * @return 转换后值
     */
    public String convertFileSize(long size)
    {
        long kb = 1024;
        long mb = kb * 1024;
        long gb = mb * 1024;
        if (size >= gb)
        {
            return String.format("%.1f GB", (float) size / gb);
        }
        else if (size >= mb)
        {
            float f = (float) size / mb;
            return String.format(f > 100 ? "%.0f MB" : "%.1f MB", f);
        }
        else if (size >= kb)
        {
            float f = (float) size / kb;
            return String.format(f > 100 ? "%.0f KB" : "%.1f KB", f);
        }
        else
        {
            return String.format("%d B", size);
        }
    }
}
