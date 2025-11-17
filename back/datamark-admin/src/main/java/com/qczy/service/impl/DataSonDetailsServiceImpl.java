package com.qczy.service.impl;

import cn.hutool.core.util.NumberUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.qczy.mapper.*;
import com.qczy.model.entity.*;
import com.qczy.model.request.DataSonQueryRequest;
import com.qczy.model.response.DataDetailsResponse;
import com.qczy.model.response.DataResponse;
import com.qczy.model.response.DataSonResponse;
import com.qczy.service.DataSonDetailsService;
import com.qczy.utils.CurrentLoginUserUtils;
import com.qczy.utils.StringUtils;
import com.qczy.utils.URLUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.util.CollectionUtils;
import org.springframework.util.ObjectUtils;

import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.math.BigDecimal;
import java.net.URLDecoder;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/12/24 10:49
 * @Description: 数据集详情
 */
@Service
public class DataSonDetailsServiceImpl implements DataSonDetailsService {

    @Autowired
    private DataSonMapper dataSonMapper;
    @Autowired
    private DataFatherMapper dataFatherMapper;
    @Autowired
    private DataImportLogMapper dataImportLogMapper;
    @Autowired
    private DictDataMapper dictDataMapper;
    @Autowired
    private MarkInfoMapper markInfoMapper;
    @Autowired
    private FileMapper fileMapper;
    @Autowired
    private LabelMapper labelMapper;
    @Autowired
    private ManyAssignMapper manyAssignMapper;
    @Autowired
    private CurrentLoginUserUtils currentLoginUserUtils;
    @Autowired
    private ManyAuditMapper manyAuditMapper;
    @Autowired
    private ManyFileMapper manyFileMapper;
    @Value("${file.accessAddress}")
    private String accessAddress;
    @Value("${upload.formalPath}")
    private String formalPath;


    @Override
    public IPage<DataResponse> getDataSetList(Page<DataResponse> pageParam, DataSonQueryRequest request) {
        //String deptIds = new CurrentLoginUserUtils().getCurrentLoginDeptIds();
        // 查出点击的字典id和子级的id
        if (request.getDataTypeId() != null) {
            List<DictDataEntity> dictDataEntityList = dictDataMapper.selectDictDataTreeSon(request.getDataTypeId());
            if (!CollectionUtils.isEmpty(dictDataEntityList)) {
                List<Integer> ids = dictDataEntityList.stream().map(DictDataEntity::getId).collect(Collectors.toList());
                StringBuilder sb = new StringBuilder();
                for (Integer id : ids) {
                    sb.append("'").append(id).append("',");
                }
                request.setDataTypeIds(sb.deleteCharAt(sb.length() - 1).toString());
            }
        }

        // 查询当前用户创建的所有数据集组
        IPage<DataResponse> fatherList = dataFatherMapper.SelectFatherResponseList(pageParam, request);
        List<DataResponse> list = fatherList.getRecords();
        if (!CollectionUtils.isEmpty(list)) {
            for (DataResponse data : list) {
                // set 二层结构
                data.setDataSonResponseList(dataSonMapper.selectDataSonByFatherId(data.getGroupId()));
                // 每个类型的总数量
                int sumCount = 0;
                // set 基础信息
                for (DataSonResponse response : data.getDataSonResponseList()) {
                    // 每个数据集的文件数量
                    sumCount += response.getCount() != null ? response.getCount() : 0;
                    // 导入记录
                    response.setDataImportCount(dataImportLogMapper.selectCount(
                            new LambdaQueryWrapper<DataImportLogEntity>()
                                    .eq(DataImportLogEntity::getSonId, response.getSonId())
                    ));
                    response.setFileSumCount(sumCount);
                }
            }
        }

        return fatherList;
    }


    @Override
    public List<DataResponse> getDataSetListNoPage() {
        // String deptIds = new CurrentLoginUserUtils().getCurrentLoginDeptIds();
        // 查询当前用户创建的所有数据集组
        List<DataResponse> fatherList = dataFatherMapper.SelectFatherResponseListNoPage();
        for (DataResponse data : fatherList) {
            // set 二层结构
            List<DataSonResponse> dataSonResponseList = dataSonMapper.selectDataSonByFatherId(data.getGroupId());
            for (DataSonResponse response : dataSonResponseList) {
                // 计算进度
                Integer count = markInfoMapper.selectCount(
                        new LambdaQueryWrapper<MarkInfoEntity>()
                                .eq(MarkInfoEntity::getSonId, response.getSonId())
                );
                if (count > 0) {
                    response.setProgress(NumberUtil.div(count.toString(), Integer.toString(response.getFileIds().split(",").length), 2)
                            .multiply(BigDecimal.valueOf(100)).intValue());
                }

            }
            data.setDataSonResponseList(dataSonResponseList);
        }
        return fatherList;
    }


    @Override
    public IPage<DataDetailsResponse> getDataDetails(Page<DataDetailsResponse> pageParam, String sonId, Integer state, Integer labelId,
                                                     Integer markUserId, Integer taskId, String sign) {
        try {
            DataSonEntity dataSonEntity = dataSonMapper.selectOne(
                    new LambdaQueryWrapper<DataSonEntity>()
                            .eq(DataSonEntity::getSonId, sonId)
            );
            if (ObjectUtils.isEmpty(dataSonEntity)) {
                throw new RuntimeException("后端异常，数据集不存在！");
            }

            if (StringUtils.isEmpty(dataSonEntity.getFileIds())) {
                return null;
            }

            // 判断是否是多人标注
            String fileIds = null;
            // markUserId == null ? dataSonEntity.getFileIds() : manyAssignMapper.selectById(markUserId).getAssignFileIds();
            if (markUserId == null) {
                fileIds = dataSonEntity.getFileIds();
            } else {
                if (StringUtils.isEmpty(sign)) {
                    fileIds = manyAssignMapper.selectById(markUserId).getAssignFileIds();
                } else {
                    ManyAuditEntity auditEntity = manyAuditMapper.selectById(markUserId);
                    fileIds = getFileIds(auditEntity.getManyMarkId(), state, auditEntity.getId());
                }
            }
            if (StringUtils.isEmpty(fileIds)) {
                return null;
            }

            IPage<DataDetailsResponse> list = null;
            switch (state) {
                case 0:      // 0.全部数据
                    if (taskId != null) {
                        fileIds = getFileIds(taskId, 0, markUserId);
                    }
                    list = fileMapper.selectFileAndlabel(pageParam, fileIds, labelId);
                    if (!CollectionUtils.isEmpty(list.getRecords())) {
                        // 进行遍历
                        for (DataDetailsResponse response : list.getRecords()) {
                            response.setSonId(dataSonEntity.getSonId());
                            response.setVersion(dataSonEntity.getVersion());
                            seLabelsName(response);

                            if (response.getMarkFileId() != null && response.getMarkFileId() > 0) {
                                // 有标记
                                FileEntity fileEntity = fileMapper.selectById(response.getMarkFileId());
                                FileEntity fileEntity1 = fileMapper.selectById(response.getFileId());
                                fileEntity.setFdName(fileEntity1.getFdName());
                                setImgPath(response, fileEntity);
                            } else {
                                // 无标记
                                FileEntity fileEntity = fileMapper.selectById(response.getFileId());
                                setImgPath(response, fileEntity);
                            }
                        }
                    }
                    break;

                case 1:    // 1.有标注信息
                    if (taskId != null) {
                        fileIds = getFileIds(taskId, 1, markUserId);
                    }


                    if (StringUtils.isEmpty(sign)) {
                        list = fileMapper.selectFileAndlabelYesMark(pageParam, fileIds, labelId);
                    } else {
                        list = fileMapper.selectFileAndlabel(pageParam, fileIds, labelId);
                    }
                    if (!CollectionUtils.isEmpty(list.getRecords())) {
                        // 进行遍历
                        for (DataDetailsResponse response : list.getRecords()) {
                            response.setSonId(dataSonEntity.getSonId());
                            response.setVersion(dataSonEntity.getVersion());
                            seLabelsName(response);
                            FileEntity fileEntity1 = fileMapper.selectById(response.getFileId());
                            if (response.getMarkFileId() != null && response.getMarkFileId() > 0) {
                                // 有标记
                                FileEntity fileEntity = fileMapper.selectById(response.getMarkFileId());
                                fileEntity.setFdName(fileEntity1.getFdName());
                                setImgPath(response, fileEntity);
                            } else {
                                response.setImgPath(fileEntity1.getHttpFilePath());
                                setImgPath(response, fileEntity1);
                            }

                        }
                    }
                    break;

                case 2:  // 无标注信息
                    if (taskId != null) {
                        fileIds = getFileIds(taskId, 2, markUserId);
                    }
                    if (StringUtils.isEmpty(sign)) {
                        list = fileMapper.selectFileAndlabelNoMark(pageParam, fileIds, labelId);
                    } else {
                        list = fileMapper.selectFileAndlabel(pageParam, fileIds, labelId);
                    }
                    if (!CollectionUtils.isEmpty(list.getRecords())) {
                        // 进行遍历
                        for (DataDetailsResponse response : list.getRecords()) {
                            response.setSonId(dataSonEntity.getSonId());
                            response.setVersion(dataSonEntity.getVersion());
                            seLabelsName(response);
                            // 无标记
                            FileEntity fileEntity1 = fileMapper.selectById(response.getFileId());
                            if (response.getMarkFileId() != null && response.getMarkFileId() > 0) {
                                // 有标记
                                FileEntity fileEntity = fileMapper.selectById(response.getMarkFileId());
                                fileEntity.setFdName(fileEntity1.getFdName());
                                setImgPath(response, fileEntity);
                            } else {
                                response.setImgPath(fileEntity1.getHttpFilePath());
                                setImgPath(response, fileEntity1);
                            }
                        }
                    }
                    break;
                case 3:  // 无效数据
                    list = fileMapper.selectFileInvalidData(pageParam, fileIds, labelId);
                    if (!CollectionUtils.isEmpty(list.getRecords())) {
                        // 进行遍历
                        for (DataDetailsResponse response : list.getRecords()) {
                            response.setSonId(dataSonEntity.getSonId());
                            response.setVersion(dataSonEntity.getVersion());
                            seLabelsName(response);
                            // 无标记
                            FileEntity fileEntity1 = fileMapper.selectById(response.getFileId());
                            if (response.getMarkFileId() != null && response.getMarkFileId() > 0) {
                                // 有标记
                                FileEntity fileEntity = fileMapper.selectById(response.getMarkFileId());
                                fileEntity.setFdName(fileEntity1.getFdName());
                                setImgPath(response, fileEntity);
                            } else {
                                response.setImgPath(fileEntity1.getHttpFilePath());
                                setImgPath(response, fileEntity1);
                            }
                        }
                    }
                    break;
            }
            return list;

        } catch (RuntimeException e) {
            throw new RuntimeException(e);
        }

    }


    @Override
    public IPage<DataDetailsResponse> getDataDetailsNoMarkFilePath(Page<DataDetailsResponse> pageParam, String sonId, Integer state, Integer markUserId, Integer taskId, String sign) {
        try {
            // 检查数据集是否存在
            DataSonEntity dataSonEntity = dataSonMapper.selectOne(
                    new LambdaQueryWrapper<DataSonEntity>()
                            .eq(DataSonEntity::getSonId, sonId)
            );
            if (ObjectUtils.isEmpty(dataSonEntity)) {
                throw new RuntimeException("后端异常，数据集不存在！");
            }

            // 获取文件id
            String fileIds = getFileIds(dataSonEntity, taskId, markUserId);

            if (StringUtils.isEmpty(fileIds)) {
                return null;
            }

            // 根据状态查询数据
            IPage<DataDetailsResponse> list = getPageDataByState(pageParam, state, taskId, fileIds, markUserId, sign);
            if (list != null && !CollectionUtils.isEmpty(list.getRecords())) {
                putPrice(dataSonEntity, list);
            }
            return list;

        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    // 优化数据
    public static Map<String, Object> paginateString(String input, long pageNumber, long pageSize) {
        // 按逗号拆分字符串
        String[] elements = input.split(",");
        List<String> elementList = Arrays.asList(elements);
        int total = elementList.size();

        // 计算总页数
        long pages = ((total + pageSize - 1) / pageSize);

        // 计算起始索引
        long startIndex = (pageNumber - 1) * pageSize;
        // 处理起始索引超出范围的情况
        if (startIndex >= total) {
            Map<String, Object> result = new HashMap<>();
            result.put("data", "");
            result.put("pages", pages);
            return result;
        }

        // 计算结束索引
        long endIndex = Math.min(startIndex + pageSize, total);

        // 由于 subList 方法接受 int 类型的参数，需要进行类型转换
        List<String> pageList = elementList.subList((int) startIndex, (int) endIndex);

        // 将列表元素用逗号拼接成字符串
        String data = String.join(",", pageList);

        Map<String, Object> result = new HashMap<>();
        result.put("data", data);
        result.put("pages", pages);
        return result;
    }

    private String getFileIds(DataSonEntity dataSonEntity, Integer taskId, Integer markUserId) {
        if (taskId == null) {
            return markUserId == null ? dataSonEntity.getFileIds() : manyAssignMapper.selectById(markUserId).getAssignFileIds();
        } else {
            return markUserId == null ? dataSonEntity.getFileIds() : manyAuditMapper.selectById(markUserId).getAuditFileIds();
        }

    }

    private IPage<DataDetailsResponse> getPageDataByState(Page<DataDetailsResponse> pageParam, Integer state, Integer taskId, String fileIds, Integer markUserId, String sign) {

        if (StringUtils.isEmpty(sign)) {
            if (taskId != null && state != 3) {
                fileIds = getFileIds(taskId, state, markUserId);
                if (StringUtils.isEmpty(fileIds)) {
                    return null;
                }
            }
            switch (state) {
                case 0:
                    return fileMapper.selectFileAndlabel(pageParam, fileIds, null);
                case 1:
                    return taskId != null ? fileMapper.selectFileAndlabel(pageParam, fileIds, null) : fileMapper.selectFileAndlabelYesMark(pageParam, fileIds, null);
                case 2:
                    return taskId != null ? fileMapper.selectFileAndlabel(pageParam, fileIds, null) : fileMapper.selectFileAndlabelNoMark(pageParam, fileIds, null);
                case 3:
                    return fileMapper.selectFileInvalidData(pageParam, fileIds, null);
                default:
                    return null;
            }
        } else {
            return fileMapper.selectFileMarkInfoAndlabel(pageParam, fileIds, null, taskId);
        }
    }

    public String getFileIds(Integer taskId, Integer state, Integer id) {
        ManyAuditEntity auditEntity = manyAuditMapper.selectById(id);
        if (ObjectUtils.isEmpty(auditEntity)) {
            return null;
        }
        List<String> idList = Arrays.asList(auditEntity.getAuditFileIds().split(","));
        List<ManyFileEntity> didNotList = manyFileMapper.selectList(
                new LambdaQueryWrapper<ManyFileEntity>()
                        .eq(ManyFileEntity::getTaskId, taskId)
                        .eq(ManyFileEntity::getIsApprove, state)
                        .eq(ManyFileEntity::getUserId, auditEntity.getMarkUserId())
                        .eq(ManyFileEntity::getAuditUserId, auditEntity.getUserId())
                        .in(ManyFileEntity::getFileId, idList)
        );

        if (CollectionUtils.isEmpty(didNotList)) {
            return null;
        }
        // 提取 fileId 并使用逗号分割
        return didNotList.stream()
                .map(ManyFileEntity::getFileId)
                .map(String::valueOf)
                .collect(Collectors.joining(","));

    }


    @Override
    public int selectFileAndlabelCount(String sonId, Integer markUserId, Integer taskId, String sign) {
        DataSonEntity dataSonEntity = dataSonMapper.selectOne(
                new LambdaQueryWrapper<DataSonEntity>()
                        .eq(DataSonEntity::getSonId, sonId)
        );
        if (ObjectUtils.isEmpty(dataSonEntity)) {
            throw new RuntimeException("后端异常，数据集不存在！");
        }

        if (StringUtils.isEmpty(dataSonEntity.getFileIds())) {
            return 0;
        }
        if (taskId != null) {
            return StringUtils.isEmpty(getFileIds(taskId, 0, markUserId)) ? 0 : getFileIds(taskId, 0, markUserId).split(",").length;
        }
        String fieIds = null;
        //String fieIds = markUserId == null ? dataSonEntity.getFileIds() : manyAssignMapper.selectById(markUserId).getAssignFileIds();
        if (markUserId == null) {
            fieIds = dataSonEntity.getFileIds();
        } else {
            if (!StringUtils.isEmpty(sign) && sign.equals("audit")) {
                ManyAuditEntity auditEntity = manyAuditMapper.selectById(markUserId);
                return StringUtils.isEmpty(getFileIds(auditEntity.getManyMarkId(), 0, markUserId)) ? 0
                        : getFileIds(auditEntity.getManyMarkId(), 0, markUserId).split(",").length;
            } else {
                fieIds = manyAssignMapper.selectById(markUserId).getAssignFileIds();
            }

        }
        if (StringUtils.isEmpty(fieIds)) {
            return 0;
        }
        return fileMapper.selectFileAndlabelCount(fieIds);
    }

    @Override
    public int selectFileAndlabelYesMarkCount(String sonId, Integer markUserId, Integer taskId, String sign) {
        DataSonEntity dataSonEntity = dataSonMapper.selectOne(
                new LambdaQueryWrapper<DataSonEntity>()
                        .eq(DataSonEntity::getSonId, sonId)
        );
        if (ObjectUtils.isEmpty(dataSonEntity)) {
            throw new RuntimeException("后端异常，数据集不存在！");
        }

        if (StringUtils.isEmpty(dataSonEntity.getFileIds())) {
            return 0;
        }
        if (taskId != null) {
            return StringUtils.isEmpty(getFileIds(taskId, 1, markUserId)) ? 0 : getFileIds(taskId, 1, markUserId).split(",").length;
        }
        //String fieIds = markUserId == null ? dataSonEntity.getFileIds() : manyAssignMapper.selectById(markUserId).getAssignFileIds();
        String fieIds = null;
        if (markUserId == null) {
            fieIds = dataSonEntity.getFileIds();
        } else {
            if (!StringUtils.isEmpty(sign) && sign.equals("audit")) {
                ManyAuditEntity auditEntity = manyAuditMapper.selectById(markUserId);
                return StringUtils.isEmpty(getFileIds(auditEntity.getManyMarkId(), 1, markUserId)) ? 0
                        : getFileIds(auditEntity.getManyMarkId(), 1, markUserId).split(",").length;
            } else {
                fieIds = manyAssignMapper.selectById(markUserId).getAssignFileIds();
            }
        }
        if (StringUtils.isEmpty(fieIds)) {
            return 0;
        }
        return fileMapper.selectFileAndlabelYesMarkCount(fieIds);
    }

    @Override
    public int selectFileAndlabelNoMarkCount(String sonId, Integer markUserId, Integer taskId, String sign) {
        DataSonEntity dataSonEntity = dataSonMapper.selectOne(
                new LambdaQueryWrapper<DataSonEntity>()
                        .eq(DataSonEntity::getSonId, sonId)
        );
        if (ObjectUtils.isEmpty(dataSonEntity)) {
            throw new RuntimeException("后端异常，数据集不存在！");
        }

        if (StringUtils.isEmpty(dataSonEntity.getFileIds())) {
            return 0;
        }
        if (taskId != null) {
            return StringUtils.isEmpty(getFileIds(taskId, 2, markUserId)) ? 0 : getFileIds(taskId, 2, markUserId).split(",").length;
        }
        String fieIds = null;
        if (markUserId == null) {
            fieIds = dataSonEntity.getFileIds();
        } else {
            if (!StringUtils.isEmpty(sign) && sign.equals("audit")) {
                ManyAuditEntity auditEntity = manyAuditMapper.selectById(markUserId);
                return StringUtils.isEmpty(getFileIds(auditEntity.getManyMarkId(), 2, markUserId)) ? 0
                        : getFileIds(auditEntity.getManyMarkId(), 2, markUserId).split(",").length;
            } else {
                fieIds = manyAssignMapper.selectById(markUserId).getAssignFileIds();
            }
        }
        if (StringUtils.isEmpty(fieIds)) {
            return 0;
        }
        return fileMapper.selectFileAndlabelNoMarkCount(fieIds);
    }

    @Override
    public int selectFileInvalidDataCount(String sonId, Integer markUserId, Integer taskId, String sign) {
        DataSonEntity dataSonEntity = dataSonMapper.selectOne(
                new LambdaQueryWrapper<DataSonEntity>()
                        .eq(DataSonEntity::getSonId, sonId)
        );
        if (ObjectUtils.isEmpty(dataSonEntity)) {
            throw new RuntimeException("后端异常，数据集不存在！");
        }

        if (StringUtils.isEmpty(dataSonEntity.getFileIds())) {
            return 0;
        }

        String fieIds = null;
        if (markUserId == null) {
            fieIds = dataSonEntity.getFileIds();
        } else {
            if (!StringUtils.isEmpty(sign) && sign.equals("audit")) {
                fieIds = manyAuditMapper.selectById(markUserId).getAuditFileIds();
            } else {
                fieIds = manyAssignMapper.selectById(markUserId).getAssignFileIds();
            }
        }
        return fileMapper.selectFileInvalidDataCount(fieIds);
    }


    // 写入标签名称
    public void seLabelsName(DataDetailsResponse response) {
        if (!StringUtils.isEmpty(response.getLabels()) && !response.getLabels().equals("0")) {
            //  List<Integer> intClusterIds = Arrays.stream(response.getLabels().split(",")).mapToInt(Integer::parseInt).boxed().collect(Collectors.toList());
            //  String[] split = response.getLabels().split(",");
            List<String> intClusterIds = Arrays.stream(response.getLabels().split(",")).collect(Collectors.toList());
            List<LabelEntity> labelEntityList = labelMapper.selectBatchIds(intClusterIds);
            if (CollectionUtils.isEmpty(labelEntityList)) {
                return;
            }
            response.setLabels(labelEntityList.stream().map(LabelEntity::getLabelName).collect(Collectors.joining(",")));
        }
    }


    public void setImgPath(DataDetailsResponse response, FileEntity fileEntity) {
        try {
            String prefixToRemove = formalPath;
            // 使用 replaceFirst 方法去掉前缀
            //  response.setImgPath(URLDecoder.decode(accessAddress + fileEntity.getFdPath().replaceFirst("^" + prefixToRemove, ""),"UTF-8"));
            if (!StringUtils.isEmpty(fileEntity.getHttpFilePath())) {
                response.setImgPath(URLUtils.encodeURL(URLDecoder.decode(accessAddress + fileEntity.getHttpFilePath().replaceFirst("^" + prefixToRemove, ""), "UTF-8")));
                response.setPreviewImgPath(URLUtils.encodeURL(URLDecoder.decode(accessAddress + fileEntity.getFdPath().replaceFirst("^" + prefixToRemove, ""), "UTF-8")));
                response.setFileName(fileEntity.getFdName());
            }


        } catch (UnsupportedEncodingException e) {
            throw new RuntimeException(e);
        }
    }


    public void putPrice(DataSonEntity dataSonEntity, IPage<DataDetailsResponse> list) throws IOException {
        String prefixToRemove = formalPath;
        // 进行遍历
        for (DataDetailsResponse response : list.getRecords()) {
            response.setSonId(dataSonEntity.getSonId());
            response.setVersion(dataSonEntity.getVersion());
            // 原始图
            FileEntity fileEntity = fileMapper.selectById(response.getFileId());
            response.setImgPath(URLUtils.encodeURL(URLDecoder.decode(accessAddress + fileEntity.getFdPath().replaceFirst("^" + prefixToRemove, ""), "UTF-8")));
            response.setPreviewImgPath(URLUtils.encodeURL(URLDecoder.decode(accessAddress + fileEntity.getHttpFilePath().replaceFirst("^" + prefixToRemove, ""), "UTF-8")));
            response.setFileName(fileEntity.getFdName());
        }
    }


}
