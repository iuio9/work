package com.qczy.service.impl;

import cn.hutool.json.JSONUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.qczy.common.markInfo.JsonConverterService;
import com.qczy.mapper.*;
import com.qczy.model.entity.*;
import com.qczy.model.request.SavaResultRequest;
import com.qczy.service.DataResultService;
import com.qczy.service.FileThreadUploadService;
import com.qczy.utils.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;
import org.springframework.util.CollectionUtils;
import org.springframework.util.ObjectUtils;

import javax.imageio.ImageIO;
import java.awt.*;
import java.io.File;
import java.io.IOException;
import java.util.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.List;
import java.util.stream.Collectors;

/**
 * 数据结果服务实现类
 * 处理数据标注结果的保存、文件复制和标签管理
 */
@Service
public class DataResultServiceImpl implements DataResultService {

    @Autowired
    private FileMapper fileMapper;
    @Autowired
    private AlgorithmTaskMapper algorithmTaskMapper;
    @Autowired
    private DataSonMapper dataSonMapper;
    @Autowired
    private DataFatherMapper dataFatherMapper;
    @Autowired
    private FileThreadUploadService fileThreadUploadService;
    @Autowired
    private MyHaoWebSocketUtils myHaoWebSocketUtils;
    @Autowired
    private FileDownloadUtils fileDownloadUtils;
    @Autowired
    private MarkInfoMapper markInfoMapper;
    @Autowired
    private DictDataMapper dictDataMapper;
    @Autowired
    private DataSonLabelMapper dataSonLabelMapper;
    @Autowired
    private LabelMapper labelMapper;

    @Value("${upload.formalPath}")
    private String formalPath;

    @Override
    public Map<String, Object> savaResult(SavaResultRequest request) {
        Map<String, Object> result = new HashMap<>();
        DataSonEntity dataSonEntity = null;
        try {
            AlgorithmTaskEntity taskEntity = algorithmTaskMapper.selectById(request.getTaskId());
            if (taskEntity == null) {
                result.put("status", 0);
                return result;
            }
            dataSonEntity = dataSonMapper.selectOne(
                    new LambdaQueryWrapper<DataSonEntity>()
                            .eq(DataSonEntity::getSonId, taskEntity.getDataSetId())
            );
            if (dataSonEntity == null) {
                result.put("status", 0);
                return result;
            }

            if (request.getType() == 2) {
                // 创建新的数据集
                Integer userId = new CurrentLoginUserUtils().getCurrentLoginUserId();
                String fatherId = String.valueOf(new SnowflakeIdWorker(0, 0).nextId());
                DataFatherEntity dataFatherEntity = savaDataFather(fatherId, request.getGroupName(), userId, request.getDataTypeId());
                if (dataFatherEntity == null) {
                    throw new RuntimeException("新增父类表失败！");
                }
                DataSonEntity dataSonEntity1 = new DataSonEntity();
                dataSonEntity1.setFatherId(fatherId);
                dataSonEntity1.setSonId(String.valueOf(new SnowflakeIdWorker(0, 0).nextId()));
                dataSonEntity1.setCreateTime(new Date());
                dataSonEntity1.setVersion(1);
                dataSonEntity1.setIsMany(0);
                dataSonEntity1.setAnoType(request.getAnoType());
                dataSonEntity1.setIsSocket(1);
                dataSonEntity1.setUserId(userId);
                int result1 = dataSonMapper.insert(dataSonEntity1);
                if (result1 == 0) {
                    throw new RuntimeException("新增子表失败！");
                }
                dataSonEntity = dataSonEntity1;
            }

            asyncSavaResult(taskEntity, dataSonEntity, request);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        result.put("status", 1);
        result.put("groupId", dataSonEntity.getFatherId());
        result.put("sonId", dataSonEntity.getSonId());
        if (request.getDataTypeId() != null) {
            result.put("dataTypeId", request.getDataTypeId());
            result.put("dictLabel", dictDataMapper.selectById(request.getDataTypeId()).getDictLabel());
        } else {
            Integer dataTypeId = dataFatherMapper.selectOne(
                    new LambdaQueryWrapper<DataFatherEntity>().eq(DataFatherEntity::getGroupId, dataSonEntity.getFatherId())
            ).getDataTypeId();
            result.put("dataTypeId", dataTypeId);
            result.put("dictLabel", dictDataMapper.selectById(dataTypeId).getDictLabel());
        }

        return result;
    }

    @Async
    public void asyncSavaResult(AlgorithmTaskEntity taskEntity,
                                DataSonEntity dataSonEntity,
                                SavaResultRequest request) {
        List<FileEntity> fileEntityList = null;
        if (StringUtils.isEmpty(request.getFileIds())) {
            fileEntityList = fileMapper.selectList(
                    new LambdaQueryWrapper<FileEntity>()
                            .eq(FileEntity::getTaskId, taskEntity.getTaskId())
            );
        }
        if (CollectionUtils.isEmpty(fileEntityList)) {
            return;
        }

        int sumCount = fileEntityList.size();
        int currentCount = 0;
        List<Integer> imageFileIdList = new ArrayList<>();
        List<List<FileEntity>> list = groupFiles(fileEntityList);
        for (List<FileEntity> fileEntities : list) {
            for (FileEntity fileEntity : fileEntities) {
                Integer imageFileId = copyAndSavaFile(fileEntity, dataSonEntity, fileEntities);
                if (imageFileId != null) {
                    imageFileIdList.add(imageFileId);
                }
                currentCount++;
                myHaoWebSocketUtils.sendMessage(dataSonEntity.getSonId(), sumCount, currentCount);
            }
        }
        int size = imageFileIdList.size();
        dataSonEntity.setStatus("100% (" + size + "/" + size + ")");
        dataSonEntity.setFileIds(
                imageFileIdList.stream()
                        .map(String::valueOf)
                        .collect(Collectors.joining(","))
        );
        dataSonMapper.updateById(dataSonEntity);
    }

    public DataFatherEntity savaDataFather(String fatherId, String groupName, Integer userId, Integer dataTypeId) {
        DataFatherEntity dataFather = new DataFatherEntity();
        dataFather.setGroupId(fatherId);
        dataFather.setGroupName(groupName);
        dataFather.setUserId(userId);
        dataFather.setCreateTime(new Date());
        dataFather.setDataTypeId(dataTypeId);
        dataFatherMapper.insert(dataFather);
        return dataFather;
    }

    public static List<List<FileEntity>> groupFiles(List<FileEntity> fileEntityList) {
        Map<String, List<FileEntity>> groupMap = new HashMap<>();
        for (FileEntity fileEntity : fileEntityList) {
            String fileName = fileEntity.getFdName();
            int dotIndex = fileName.lastIndexOf('.');
            if (dotIndex != -1) {
                String baseName = fileName.substring(0, dotIndex);
                groupMap.computeIfAbsent(baseName, k -> new ArrayList<>()).add(fileEntity);
            }
        }

        for (List<FileEntity> group : groupMap.values()) {
            group.sort((file1, file2) -> {
                String ext1 = getFileExtension(file1.getFdName());
                String ext2 = getFileExtension(file2.getFdName());

                if ("json".equals(ext1) && !"json".equals(ext2)) {
                    return 1;
                } else if (!"json".equals(ext1) && "json".equals(ext2)) {
                    return -1;
                }
                return 0;
            });
        }

        return new ArrayList<>(groupMap.values());
    }

    private static String getFileExtension(String fileName) {
        int dotIndex = fileName.lastIndexOf('.');
        return dotIndex != -1 ? fileName.substring(dotIndex + 1).toLowerCase() : "";
    }

    @Autowired
    private JsonConverterService jsonConverterService;

    public Integer copyAndSavaFile(FileEntity fileEntity, DataSonEntity dataSon, List<FileEntity> fileEntities) {
        Integer newFileId = null;
        String sourcePath = fileEntity.getFdPath();
        String destinationPath = formalPath + dataSon.getFatherId() + "/" + "v" + dataSon.getVersion() + "/" + "source" + "/";
        Path source = Paths.get(sourcePath);
        Path destination = Paths.get(destinationPath + fileEntity.getFdName());

        FileEntity fileEntity1 = new FileEntity();
        String shrinkPath = null;

        if (ImageUtils.isImage(fileEntity.getFdPath())) {
            try {
                File file = new File(destinationPath);
                if (!file.exists()) {
                    file.mkdirs();
                }
                Files.copy(source, destination, StandardCopyOption.REPLACE_EXISTING);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }

            String originalFilename = fileEntity.getFdName();
            String safeFilename = processSpecialChars(originalFilename);

            fileEntity1.setFdName(safeFilename);
            fileEntity1.setFdPath(destinationPath + safeFilename);

            // 传递三个参数调用saveShrinkFile
            shrinkPath = FileThreadUploadServiceImpl.saveShrinkFile(
                    formalPath + dataSon.getFatherId() + "/" + "v" + dataSon.getVersion(),
                    fileEntity1.getFdPath(),
                    safeFilename
            );

            fileEntity1.setHttpFilePath(shrinkPath);
            try {
                Image image = ImageIO.read(new File(fileEntity1.getFdPath()));
                fileEntity1.setWidth(image.getWidth(null));
                fileEntity1.setHeight(image.getHeight(null));
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
            fileEntity1.setFileStatus(0);
            fileEntities.get(0).setWidth(fileEntity1.getWidth());
            fileEntities.get(0).setHeight(fileEntity1.getHeight());
            fileMapper.insert(fileEntity1);
            newFileId = fileEntity1.getId();
            fileEntities.get(0).setId(newFileId);
        } else {
            FileEntity entity = fileEntities.get(0);
            MarkInfoEntity markInfoEntity = new MarkInfoEntity();
            markInfoEntity.setSonId(dataSon.getSonId());
            markInfoEntity.setFileId(entity.getId());
            markInfoEntity.setMarkFileId(entity.getId());
            markInfoEntity.setWidth(fileEntities.get(0).getWidth());
            markInfoEntity.setOperateWidth(fileEntities.get(0).getWidth());
            markInfoEntity.setHeight(fileEntities.get(0).getHeight());
            markInfoEntity.setOperateHeight(fileEntities.get(0).getHeight());

            String jsonStr = FileToStringUtils.readTextFile(fileEntity.getFdPath());
            markInfoEntity.setLabelMarkInfo(jsonStr);
            LabelmeImageData bean = JSONUtil.toBean(jsonStr, LabelmeImageData.class);
            List<WebRectangleShape> webRectangleShapes = FormatConverter.convertLabelmeDataToWeb(bean, markInfoEntity);

            String labelIds = setDataSetLabel(webRectangleShapes, dataSon.getSonId());
            if (!StringUtils.isEmpty(labelIds)) {
                markInfoEntity.setLabels(labelIds);
            }

            try {
                markInfoEntity.setMarkInfo(jsonConverterService.convertPreToNext(fileEntity.getFdPath()));
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
            markInfoMapper.insert(markInfoEntity);
            fileDownloadUtils.writeFile(markInfoEntity);
        }

        return newFileId;
    }

    private String setDataSetLabel(List<WebRectangleShape> webRectangleShapes, String sonId) {
        if (CollectionUtils.isEmpty(webRectangleShapes)) {
            return "";
        }
        StringBuilder labelIds = new StringBuilder();
        for (WebRectangleShape webRectangleShape : webRectangleShapes) {
            if (ObjectUtils.isEmpty(webRectangleShape.getProps())) {
                continue;
            }
            if (StringUtils.isEmpty(webRectangleShape.getProps().getName())) {
                continue;
            }

            if (dataSonLabelMapper.selectBySonIdAndLabelNameCount(sonId, webRectangleShape.getProps().getName()) > 0) {
                continue;
            }

            LabelEntity labelEntity = new LabelEntity();
            labelEntity.setLabelName(webRectangleShape.getProps().getName());
            labelEntity.setLabelColor("#7C0DDD");
            labelEntity.setLabelGroupId(0);
            labelEntity.setCreateTime(new Date());
            labelMapper.insert(labelEntity);

            DataSonLabelEntity dataSonLabelEntity = new DataSonLabelEntity();
            dataSonLabelEntity.setSonId(sonId);
            dataSonLabelEntity.setLabelId(labelEntity.getId());
            dataSonLabelMapper.insert(dataSonLabelEntity);
            labelIds.append(labelEntity.getId()).append(",");
        }
        if (!StringUtils.isEmpty(labelIds.toString())) {
            labelIds.deleteCharAt(labelIds.length() - 1);
        }
        return labelIds.toString();
    }

    /**
     * 处理文件名中的+和-特殊字符
     */
    private String processSpecialChars(String filename) {
        if (filename == null) {
            return UUID.randomUUID().toString() + ".tmp";
        }
        // 将+和-替换为下划线
        return filename.replaceAll("[+-]", "_");
    }
}