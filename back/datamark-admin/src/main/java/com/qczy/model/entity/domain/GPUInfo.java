package com.qczy.model.entity.domain;
public class GPUInfo {
    private String name;
    private String vendor;
    private String deviceId;
    private String driverVersion;
    private long vRamTotal; // 总显存大小
    private long vRamUsed;  // 已用显存
    private double coreUsage; // 核心使用率

    // 构造函数
    public GPUInfo(String name, String vendor, String deviceId, String driverVersion, long vRamTotal, long vRamUsed, double coreUsage) {
        this.name = name;
        this.vendor = vendor;
        this.deviceId = deviceId;
        this.driverVersion = driverVersion;
        this.vRamTotal = vRamTotal;
        this.vRamUsed = vRamUsed;
        this.coreUsage = coreUsage;
    }

    // Getter 和 Setter 方法
    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getVendor() {
        return vendor;
    }

    public void setVendor(String vendor) {
        this.vendor = vendor;
    }

    public String getDeviceId() {
        return deviceId;
    }

    public void setDeviceId(String deviceId) {
        this.deviceId = deviceId;
    }

    public String getDriverVersion() {
        return driverVersion;
    }

    public void setDriverVersion(String driverVersion) {
        this.driverVersion = driverVersion;
    }

    public long getVRamTotal() {
        return vRamTotal;
    }

    public void setVRamTotal(long vRamTotal) {
        this.vRamTotal = vRamTotal;
    }

    public long getVRamUsed() {
        return vRamUsed;
    }

    public void setVRamUsed(long vRamUsed) {
        this.vRamUsed = vRamUsed;
    }

    public double getCoreUsage() {
        return coreUsage;
    }

    public void setCoreUsage(double coreUsage) {
        this.coreUsage = coreUsage;
    }

    // 打印信息的方法
    public void printInfo() {
        System.out.println("GPU 名称: " + name);
        System.out.println("GPU 制造商: " + vendor);
        System.out.println("GPU 设备ID: " + deviceId);
        System.out.println("GPU 驱动版本: " + driverVersion);
        System.out.println("GPU 总显存: " + vRamTotal + " MB");
        System.out.println("GPU 已用显存: " + vRamUsed  + " MB");
        System.out.println("GPU 核心使用率: " + coreUsage + " %");
        System.out.println("------------------------------------------------");
    }
}
