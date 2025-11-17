package com.qczy.model.entity;

import com.baomidou.mybatisplus.annotation.FieldFill;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import com.fasterxml.jackson.annotation.JsonFormat;
import com.qczy.common.base.BaseEntity;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiModelProperty;
import lombok.Data;
import org.springframework.format.annotation.DateTimeFormat;

import java.io.Serializable;
import java.util.Date;

@Data
@TableName("qczy_system_status")
public class SystemStatusEntity  implements Serializable{

    private static final long serialVersionUID = 1L;



    @ApiModelProperty(value = "创建时间")
    @DateTimeFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss", timezone = "GMT+8")
    @TableField(fill = FieldFill.INSERT,value = "create_time")
    private Date createTime;


    @ApiModelProperty(value = "更新时间")
    @DateTimeFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss", timezone = "GMT+8")
    @TableField(value="update_time", fill = FieldFill.INSERT_UPDATE)
    private Date updateTime;
    @TableId
    private Long id;

    @ApiModelProperty("获取的类型(包含算法 0 和存储 1)")
    private String type;

    // GPU
    @ApiModelProperty("gpu总计")
    private String gpuTotal;
    @ApiModelProperty("gpu使用HZ")
    private String gpuUsed;
    @ApiModelProperty("空闲gpu")
    private String gpuFree;
    @ApiModelProperty("gpu使用率")
    private String gpuUsage;

    // CPU
    @ApiModelProperty("总计cpu大小")
    private String cpuTotal;
    @ApiModelProperty("cpu使用大小")
    private String cpuUsed;
    @ApiModelProperty("cpu剩余值")
    private String cpuFree;
    @ApiModelProperty("cpu使用率")
    private String cpuUsage;

    // Memory
    @ApiModelProperty("内存总计")
    private String memTotal;
    @ApiModelProperty("内存使用大小(MB)")
    private String memUsed;
    @ApiModelProperty("内存剩余大小(MB)")
    private String memFree;
    @ApiModelProperty("内存使用率")
    private String memUsage;

    // System Information
    @ApiModelProperty("电脑名称")
    private String sysComputerName;
    @ApiModelProperty("设备ip")
    private String sysComputerIp;
    @ApiModelProperty("用户目录")
    private String sysUserDir;
    @ApiModelProperty("系统名称")
    private String sysOsName;
    @ApiModelProperty("系统")
    private String sysOsArch;

    // File System Information
    @ApiModelProperty("磁盘总计")
    private String sysFileinfoTotal;
    @ApiModelProperty("磁盘使用（单位G）")
    private String sysFileinfoUsed;
    @ApiModelProperty("剩余磁盘大小")
    private String sysFileinfoFree;
    @ApiModelProperty("磁盘使用率")
    private String sysFileinfoUsage;


    public SystemStatusEntity(String type,String gpuTotal, String gpuUsed, String gpuFree, String gpuUsage, String cpuTotal, String cpuUsed, String cpuFree, String cpuUsage, String memTotal, String memUsed, String memFree, String memUsage, String sysComputerName, String sysComputerIp, String sysUserDir, String sysOsName, String sysOsArch, String sysFileInfoTotal, String sysFileInfoUsed, String sysFileInfoFree, String sysFileInfoUsage) {
        this.type= type;
        this.gpuTotal = gpuTotal;
        this.gpuUsed = gpuUsed;
        this.gpuFree = gpuFree;
        this.gpuUsage = gpuUsage;
        this.cpuTotal = cpuTotal;
        this.cpuUsed = cpuUsed;
        this.cpuFree = cpuFree;
        this.cpuUsage = cpuUsage;
        this.memTotal = memTotal;
        this.memUsed = memUsed;
        this.memFree = memFree;
        this.memUsage = memUsage;
        this.sysComputerName = sysComputerName;
        this.sysComputerIp = sysComputerIp;
        this.sysUserDir = sysUserDir;
        this.sysOsName = sysOsName;
        this.sysOsArch = sysOsArch;
        this.sysFileinfoTotal = sysFileInfoTotal;
        this.sysFileinfoUsed = sysFileInfoUsed;
        this.sysFileinfoFree = sysFileInfoFree;
        this.sysFileinfoUsage = sysFileInfoUsage;
    }
}
