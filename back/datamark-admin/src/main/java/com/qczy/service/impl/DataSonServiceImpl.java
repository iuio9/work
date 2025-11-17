package com.qczy.service.impl;


import com.alibaba.fastjson.JSONObject;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.qczy.common.constant.SystemConstant;
import com.qczy.common.result.Result;
import com.qczy.handler.MyWebSocketHandler;
import com.qczy.mapper.*;
import com.qczy.model.entity.*;
import com.qczy.model.request.*;
import com.qczy.service.*;
import com.qczy.utils.CurrentLoginUserUtils;
import com.qczy.utils.MyHaoWebSocketUtils;
import com.qczy.utils.SnowflakeIdWorker;
import com.qczy.utils.StringUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.util.CollectionUtils;
import org.springframework.util.ObjectUtils;

import java.io.IOException;
import java.util.*;
import java.util.List;
import java.util.stream.Collectors;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/7 21:49
 * @Description:
 */
@Service
public class DataSonServiceImpl extends ServiceImpl<DataSonMapper, DataSonEntity> implements DataSonService {

    @Autowired
    private DataSonMapper dataSonMapper;
    @Autowired
    private DataFatherMapper dataFatherMapper;
    @Autowired
    private UserMapper userMapper;
    @Autowired
    private FileService fileService;
    @Autowired
    private FileMapper fileMapper;
    @Autowired
    private AlgorithmTaskMapper algorithmTaskMapper;
    @Autowired
    private DataMarkService dataMarkService;
    @Autowired
    private FileExtractService fileExtractService;
    @Autowired
    private LabelGroupService labelGroupService;
    @Autowired
    private MyWebSocketHandler myWebSocketHandler;
    @Autowired
    private FileThreadUploadService fileThreadUploadService;


    @Override
    public int insertDataSet(DataSonEntityRequest dataSonRequest) {

        try {
            // status：0：从临时文件拷贝，1：从正式文件拷贝 , 2:文件夹上传
            return savaDataSet(dataSonRequest,
                    dataSonRequest.getImportMode() != null && dataSonRequest.getImportMode() == 0 ? 0 : 2
                    , null);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }


    @Transactional
    @Override
    public int getResultDataSetSave(ResultDataSonRequest request) {

        try {
            DataSonEntityRequest dataSonRequest = new DataSonEntityRequest();
            dataSonRequest.setGroupName(request.getGroupName());
            dataSonRequest.setVersion(1);
            dataSonRequest.setCreateTime(new Date());
            dataSonRequest.setDataTypeId(request.getDataTypeId());

            /**
             *  获取所有的文件
             */
            List<FileEntity> fileEntityList = fileMapper.selectList(
                    new LambdaQueryWrapper<FileEntity>()
                            .eq(FileEntity::getTaskId, request.getTaskId())
            );

            if (CollectionUtils.isEmpty(fileEntityList)) {
                throw new RuntimeException("文件不存在！");
            }

            String fileIds = fileEntityList.stream()
                    .map(fileEntity -> String.valueOf(fileEntity.getId()))
                    .collect(Collectors.joining(","));
            dataSonRequest.setFileIds(fileIds);
            dataSonRequest.setIsSocket(1);

            // status：0，从临时文件拷贝，1：从正式文件拷贝
            return savaDataSet(dataSonRequest, 1, request);


        } catch (Exception e) {
            System.out.println(e.getMessage());
            throw new RuntimeException(e);
        }
    }

    // 判断任务类型是否为缺陷生成，如果是缺陷生成，则进行图片与json合并
    public void setFlawFileAndJson(ResultDataSonRequest request, DataSonEntityRequest dataSonRequest) {
        AlgorithmTaskEntity taskEntity = algorithmTaskMapper.selectById(request.getTaskId());
        // 执行json跟图片合并
        if (!ObjectUtils.isEmpty(taskEntity)) {
            if (!StringUtils.isEmpty(taskEntity.getModelId()) && taskEntity.getModelId().equals("4")) {
                dataMarkService.setMarkFileJsonWrite(dataSonRequest.getSonId());
            }
        }
    }


    // 新增父表
    public int savaDataFather(String fatherId, String groupName, Integer userId, Integer dataTypeId) {
        DataFatherEntity dataFather = new DataFatherEntity();
        dataFather.setGroupId(fatherId);
        dataFather.setGroupName(groupName);
        dataFather.setUserId(userId);
        dataFather.setCreateTime(new Date());
        dataFather.setDataTypeId(dataTypeId);
        return dataFatherMapper.insert(dataFather);
    }


    @Override
    public int addDataVersion(SaveSonVersionRequest saveSonVersionRequest) {
        try {
            // 获取登录的用户
            UserEntity user = userMapper.selectById(new CurrentLoginUserUtils().getCurrentLoginUserId());
            // 获取父表id
            if (ObjectUtils.isEmpty(user)) {
                throw new RuntimeException("后端异常，获取父类id失败！");
            }
            DataFatherEntity dataFather = dataFatherMapper.selectOne(
                    new LambdaQueryWrapper<DataFatherEntity>()
                            .eq(DataFatherEntity::getGroupId, saveSonVersionRequest.getGroupId())
            );
            if (ObjectUtils.isEmpty(dataFather)) {
                throw new RuntimeException("后端异常，获取父类对象失败！");
            }

            // 根据数据集组查询之前的版本
            DataSonEntity dataSonEntity = dataSonMapper.selectOne(
                    new LambdaQueryWrapper<DataSonEntity>()
                            .eq(DataSonEntity::getFatherId, saveSonVersionRequest.getGroupId())
                            .eq(DataSonEntity::getVersion, saveSonVersionRequest.getVersion())
            );
            if (ObjectUtils.isEmpty(dataSonEntity)) {
                throw new RuntimeException("后端异常，获取历史数据集对象失败！");
            }


            DataSonEntityRequest dataSonRequest = new DataSonEntityRequest();

            dataSonRequest.setSonId(String.valueOf(new SnowflakeIdWorker(0, 0).nextId()));
            dataSonRequest.setVersion(saveSonVersionRequest.getNewVersion());
            dataSonRequest.setUserId(user.getId());
            dataSonRequest.setCreateTime(new Date());
            dataSonRequest.setFatherId(saveSonVersionRequest.getGroupId());
            dataSonRequest.setAnoType(dataSonEntity.getAnoType());

            int result = dataSonMapper.insert(dataSonRequest);

            if (saveSonVersionRequest.getIsInherit() == 1) { // 说明是继承的，拷贝以前数据集的文件
                // 判断以前版本是否有文件id
                if (!StringUtils.isEmpty(dataSonEntity.getFileIds())) {
                    dataSonRequest.setFileIds(dataSonEntity.getFileIds());
                    //拷贝之前版本的图片
                    fileService.savaDataSonCopyFile(dataSonRequest);
                }
            } else {
                dataSonRequest.setStatus("0% (0/" + "0" + ")");
                dataSonMapper.updateById(dataSonRequest);
            }
            return result;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public int updateDataSetRemark(UpdateDataSetRequest request) {
        DataSonEntity dataSonEntity = dataSonMapper.selectOne(
                new LambdaQueryWrapper<DataSonEntity>()
                        .eq(DataSonEntity::getSonId, request.getSonId())
        );
        if (ObjectUtils.isEmpty(dataSonEntity)) {
            throw new RuntimeException("后端异常，查询不到此对象！");
        }
        dataSonEntity.setRemark(request.getRemark());
        return dataSonMapper.updateById(dataSonEntity);
    }


    public int savaDataSet(DataSonEntityRequest dataSonRequest, int status, ResultDataSonRequest request) {
        try {
            //获取到登录的用户名
            Integer userId = new CurrentLoginUserUtils().getCurrentLoginUserId();
            UserEntity user = userMapper.selectOne(
                    new LambdaQueryWrapper<UserEntity>()
                            .eq(UserEntity::getId, userId)
                            .eq(UserEntity::getIsDeleted, SystemConstant.SYSTEM_NO_DISABLE)
            );
            if (Objects.isNull(user)) {
                throw new RuntimeException("后端异常，当前未获取到用户名！");
            }

            dataSonRequest.setUserId(user.getId());

            // 首先先新增父表
            // 生成父id
            String fatherId = String.valueOf(new SnowflakeIdWorker(0, 0).nextId());
            int fatherResult = savaDataFather(fatherId, dataSonRequest.getGroupName(), userId, dataSonRequest.getDataTypeId());
            if (fatherResult < SystemConstant.MAX_SIZE) {
                throw new RuntimeException("后端异常，新增父类表失败！");
            }

            dataSonRequest.setFatherId(fatherId);
            // 睡眠100毫秒，避免id重复
            Thread.sleep(100);
            dataSonRequest.setSonId(String.valueOf(new SnowflakeIdWorker(0, 0).nextId()));
            dataSonRequest.setCreateTime(new Date());
            dataSonRequest.setIsSocket(1);
            int result = dataSonMapper.insert(dataSonRequest);


            // 判断传入的是 标签组 还是 标签
            if (!StringUtils.isEmpty(dataSonRequest.getTagSelectionMode())) {
                switch (dataSonRequest.getTagSelectionMode()) {
                    case "group":
                        if (!StringUtils.isEmpty(dataSonRequest.getGroupIds()))
                            labelGroupService.addDataSonAndLabelGroup(dataSonRequest.getSonId(), dataSonRequest.getGroupIds());

                        break;
                    case "single":
                        if (!StringUtils.isEmpty(dataSonRequest.getTagIds()))
                            labelGroupService.addDataSonAndLabelIds(dataSonRequest.getSonId(), dataSonRequest.getTagIds());
                        break;
                }
            }

            // 判断是否有文件
            // 1：没有进行上传文件
            if (StringUtils.isEmpty(dataSonRequest.getFileIds())) {
                dataSonRequest.setStatus("0% (0/" + "0" + ")");
                dataSonRequest.setIsSocket(0);
                return dataSonMapper.updateById(dataSonRequest);
            }
            // 首先先推送一个进度条
            Result<Object> result1 = new Result<>();
            result1.setCode(200);
            result1.setData(1);
            System.out.println("进度条：" + dataSonRequest.getSonId());
            myWebSocketHandler.sendMessageToUser("dataSetProgress", dataSonRequest.getSonId(), JSONObject.toJSONString(result1));
            // 2:上传文件
            savaFiles(dataSonRequest, status, request);
            // 判断是否上传的是有标注信息
        /*    if (dataSonRequest.getMarkStatus() == 1) {
                fileMarkService.addMarkSon(dataSonRequest.getSonId(),dataSonRequest.getFileIds());
            }*/

            return result;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }


    public void savaFiles(DataSonEntityRequest dataSonRequest, int status, ResultDataSonRequest request) {
        try {
            if (status == 0) {
                // fileService.savaDataTempSonCopyFile(dataSonRequest, "");
                fileThreadUploadService.savaDataTempSonCopyFile(dataSonRequest, "");
            } else if (status == 1) {
                fileService.savaDataSonCopyFile1(dataSonRequest, request);
            } else if (status == 2) {
                fileExtractService.fileExtract(dataSonRequest);
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }


}
