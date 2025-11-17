package com.qczy.service.impl;

import cn.hutool.core.util.NumberUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.service.IService;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.qczy.common.constant.SystemConstant;
import com.qczy.mapper.*;
import com.qczy.model.entity.*;
import com.qczy.model.request.DataSetImportRequest;
import com.qczy.model.request.DataSonEntityRequest;
import com.qczy.model.request.UpdateDataSetRequest;
import com.qczy.service.*;
import com.qczy.utils.CurrentLoginUserUtils;
import com.qczy.utils.StringUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.util.CollectionUtils;
import org.springframework.util.ObjectUtils;

import java.io.IOException;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/12 14:22
 * @Description:
 */
@Service
public class DataFatherServiceImpl extends ServiceImpl<DataFatherMapper,DataFatherEntity> implements DataFatherService {
    @Autowired
    private DataFatherMapper dataFatherMapper;

    @Autowired
    private DataSonMapper dataSonMapper;

    @Autowired
    private FileService fileService;

    @Autowired
    private FileMapper fileMapper;

    @Autowired
    private FileExtractService fileExtractService;

    @Autowired
    private MarkInfoMapper markInfoMapper;

    @Autowired
    private LabelGroupMapper labelGroupMapper;

    @Autowired
    private LabelMapper labelMapper;

    @Autowired
    private DataSonLabelMapper dataSonLabelMapper;

    @Autowired
    private FileThreadUploadService fileThreadUploadService;

    @Autowired
    private LabelGroupService labelGroupService;

    @Value("${upload.formalPath}")
    private String formalPath;

    @Transactional
    @Override
    public int deleteDataGroup(String groupId) throws IOException {
        List<DataSonEntity> sonList = dataSonMapper.selectList(
                new LambdaQueryWrapper<DataSonEntity>()
                        .eq(DataSonEntity::getFatherId, groupId)
        );

        // 删除文件目录，以及文件信息
        // ①删除文件目录
        fileService.deleteFile(groupId);
     /*   if (fileService.deleteFile(groupId) < SystemConstant.MAX_SIZE) {
            throw new RuntimeException("后端异常，文件目录删除失败！");
        }*/

        // ②删除文件表
        for (DataSonEntity dataSonEntity : sonList) {
            if (!StringUtils.isEmpty(dataSonEntity.getFileIds())) {
                String[] fileIds = dataSonEntity.getFileIds().split(",");
                if (fileIds.length > 0 && fileMapper.deleteFileByIds(fileIds) < SystemConstant.MAX_SIZE) {
                    throw new RuntimeException("后端异常，文件表删除失败");
                }
            }

            // 删除标注信息表
            List<MarkInfoEntity> markInfoIds = markInfoMapper.selectList(
                    new LambdaQueryWrapper<MarkInfoEntity>().eq(MarkInfoEntity::getSonId, dataSonEntity.getSonId())
            );
            if (!CollectionUtils.isEmpty(markInfoIds)) {
                markInfoMapper.deleteBatchIds(markInfoIds.stream().map(MarkInfoEntity::getId).collect(Collectors.toList()));
            }

        }


        // 删除子表
        List<Integer> list = sonList.stream().map(DataSonEntity::getId).collect(Collectors.toList());
        System.out.println("---------------------------" + list);
        if (dataSonMapper.deleteDataSonByIds(list) < SystemConstant.MAX_SIZE) {
            throw new RuntimeException("后端异常，子表删除失败！");
        }
        // 删除当前表
        return dataFatherMapper.delete(
                new LambdaQueryWrapper<DataFatherEntity>()
                        .eq(DataFatherEntity::getGroupId, groupId)
        );
    }


    @Override
    public int deleteDataSet(String sonId) throws IOException {
        DataSonEntity dataSonEntity = dataSonMapper.selectOne(
                new LambdaQueryWrapper<DataSonEntity>()
                        .eq(DataSonEntity::getSonId, sonId)
        );

        // 如果此数据集组只剩下一条，数据集组也删除
        Integer num = dataSonMapper.selectCount(
                new LambdaQueryWrapper<DataSonEntity>()
                        .eq(DataSonEntity::getFatherId, dataSonEntity.getFatherId())
        );

        if (num == 1) {
            // 进行数据集组删除
            dataFatherMapper.delete(
                    new LambdaQueryWrapper<DataFatherEntity>()
                            .eq(DataFatherEntity::getGroupId, dataSonEntity.getFatherId())
            );
            if (!ObjectUtils.isEmpty(dataSonEntity) && !StringUtils.isEmpty(dataSonEntity.getFileIds())) {
                // 删除文件目录，以及文件信息
                // ①删除文件目录
                // 拼接删除路径
                fileService.deleteFile(dataSonEntity.getFatherId());
        /*        if (fileService.deleteFile(dataSonEntity.getFatherId()) < SystemConstant.MAX_SIZE)
                    throw new RuntimeException("后端异常，文件目录删除失败！");*/

                // 删除标注信息表
                List<MarkInfoEntity> markInfoIds = markInfoMapper.selectList(
                        new LambdaQueryWrapper<MarkInfoEntity>().eq(MarkInfoEntity::getSonId, dataSonEntity.getSonId())
                );
                if (!CollectionUtils.isEmpty(markInfoIds)) {
                    markInfoMapper.deleteBatchIds(markInfoIds.stream().map(MarkInfoEntity::getId).collect(Collectors.toList()));
                }

                // ②删除文件表
                if (!StringUtils.isEmpty(dataSonEntity.getFileIds())) {
                    String[] fileIds = dataSonEntity.getFileIds().split(",");
                    if (fileIds.length > 0)
                        fileMapper.deleteFileByIds(fileIds);
                }
            }


        } else {
            if (!ObjectUtils.isEmpty(dataSonEntity) && !StringUtils.isEmpty(dataSonEntity.getFileIds())) {
                // 删除文件目录，以及文件信息
                // ①删除文件目录
                // 拼接删除路径
                fileService.deleteFile(dataSonEntity.getFatherId(), dataSonEntity.getVersion().toString());
              /*  if (fileService.deleteFile(dataSonEntity.getFatherId(), dataSonEntity.getVersion().toString()) < SystemConstant.MAX_SIZE)
                    throw new RuntimeException("后端异常，文件目录删除失败！");*/

                // 删除标注信息表
                List<MarkInfoEntity> markInfoIds = markInfoMapper.selectList(
                        new LambdaQueryWrapper<MarkInfoEntity>().eq(MarkInfoEntity::getSonId, dataSonEntity.getSonId())
                );
                if (!CollectionUtils.isEmpty(markInfoIds)) {
                    markInfoMapper.deleteBatchIds(markInfoIds.stream().map(MarkInfoEntity::getId).collect(Collectors.toList()));
                }

                // ②删除文件表
                if (!StringUtils.isEmpty(dataSonEntity.getFileIds())) {
                    String[] fileIds = dataSonEntity.getFileIds().split(",");
                    if (fileIds.length > 0)
                        fileMapper.deleteFileByIds(fileIds);

                }
            }
        }


        // 删除当前表

        return dataSonMapper.delete(
                new LambdaQueryWrapper<DataSonEntity>()
                        .eq(DataSonEntity::getSonId, sonId)
        );
    }


    // 数据集导入
    @Override
    public int dataSetImport(DataSonEntityRequest dataSonRequest) {
        try {
            DataSonEntity dataSonEntity = dataSonMapper.selectOne(
                    new LambdaQueryWrapper<DataSonEntity>()
                            .eq(DataSonEntity::getSonId, dataSonRequest.getSonId())
            );
            if (ObjectUtils.isEmpty(dataSonEntity)) {
                throw new RuntimeException("后端异常，查询数据集失败！");
            }
            //获取到登录的用户名
            Integer userId = new CurrentLoginUserUtils().getCurrentLoginUserId();
            dataSonRequest.setUserId(userId);
            dataSonRequest.setId(dataSonEntity.getId());
            dataSonEntity.setIsSocket(1);
            dataSonMapper.updateById(dataSonEntity);

            dataSonRequest.setFatherId(dataSonEntity.getFatherId());
            dataSonRequest.setSonId(dataSonEntity.getSonId());

            // 判断是否上传了标签
          //  if (!StringUtils.isEmpty(dataSonRequest.getGroupIds())) {
                // 批量删除标签
                LambdaQueryWrapper<DataSonLabelEntity> deleteWrapper = new LambdaQueryWrapper<>();
                deleteWrapper.eq(DataSonLabelEntity::getSonId, dataSonEntity.getSonId());
                dataSonLabelMapper.delete(deleteWrapper);


                // 判断传入的是 标签组 还是 标签
                if (!StringUtils.isEmpty(dataSonRequest.getTagSelectionMode())) {
                    switch (dataSonRequest.getTagSelectionMode()) {
                        case "group":
                            labelGroupService.addDataSonAndLabelGroup(dataSonRequest.getSonId(), dataSonRequest.getGroupIds());
                            break;
                        case "single":
                            labelGroupService.addDataSonAndLabelIds(dataSonRequest.getSonId(), dataSonRequest.getTagIds());
                            break;
                    }
                }

          //  }


            if (dataSonRequest.getImportMode() == null) {
                return 1;
            }
            if (StringUtils.isEmpty(dataSonRequest.getFileIds())) {
                return 1;
            }


            if (dataSonRequest.getImportMode() == 1) {  // 压缩包上传
                if (!StringUtils.isEmpty(dataSonEntity.getFileIds())) {
                    dataSonRequest.setOldFileIds(dataSonEntity.getFileIds());
                }
                fileExtractService.fileExtract(dataSonRequest);
            } else { // 普通上传
                if (!StringUtils.isEmpty(dataSonEntity.getFileIds())) {
                    // 记录之前的文件id
                    String fileIds = dataSonEntity.getFileIds();
                    // set新增的文件id
                    dataSonEntity.setFileIds(dataSonRequest.getFileIds());
                    //fileService.savaDataTempSonCopyFile(dataSonEntity, fileIds);
                    fileThreadUploadService.savaDataTempSonCopyFile(dataSonEntity,fileIds);
                } else {
                    dataSonEntity.setFileIds(dataSonRequest.getFileIds());
                    //fileService.savaDataTempSonCopyFile(dataSonEntity, "");
                    fileThreadUploadService.savaDataTempSonCopyFile(dataSonEntity,"");
                }
            }




       /*     dataSonEntity.setFileIds(request.getFileIds());
            // 进行文件上传

            if (StringUtils.isEmpty(fileIds)) {
                dataSonEntity.setFileIds(fileId);
            } else {
                dataSonEntity.setFileIds(fileIds + "," + fileId);
            }


            // 计算进度
            if (!ObjectUtils.isEmpty(dataSonEntity)) {
                String[] files = dataSonEntity.getFileIds().split(",");
                Integer count = markInfoMapper.selectCount(
                        new LambdaQueryWrapper<MarkInfoEntity>()
                                .eq(MarkInfoEntity::getSonId, dataSonEntity.getSonId())
                );
                int num = NumberUtil.div(count.toString(), Integer.toString(files.length), 2).multiply(BigDecimal.valueOf(100)).intValue();
                dataSonEntity.setStatus(num + "% " + ("(" + count + "/" + files.length + ")"));
            }*/


            return 1;
            // return dataSonMapper.updateById(dataSonEntity);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public int updateDataSetName(UpdateDataSetRequest request) {
        DataFatherEntity dataFatherEntity = dataFatherMapper.selectOne(
                new LambdaQueryWrapper<DataFatherEntity>()
                        .eq(DataFatherEntity::getGroupId, request.getGroupId())
        );
        if (ObjectUtils.isEmpty(dataFatherEntity)) {
            throw new RuntimeException("后端异常，查询不到此对象！");
        }
        dataFatherEntity.setGroupName(request.getGroupName());
        return dataFatherMapper.updateById(dataFatherEntity);
    }


}
