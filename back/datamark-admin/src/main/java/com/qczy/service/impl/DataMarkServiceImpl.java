package com.qczy.service.impl;

import cn.hutool.core.bean.BeanUtil;
import cn.hutool.core.bean.copier.CopyOptions;
import cn.hutool.core.util.NumberUtil;
import cn.hutool.json.JSONUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.qczy.common.markInfo.JsonToVocXml;
import com.qczy.common.markInfo.JsonTransformer;
import com.qczy.mapper.*;
import com.qczy.model.entity.*;
import com.qczy.model.request.DataSonQueryRequest;
import com.qczy.model.response.DataDetailsResponse;
import com.qczy.model.response.DataMarkResponse;
import com.qczy.service.DataMarkService;
import com.qczy.service.DictDataService;
import com.qczy.service.MarkFileService;
import com.qczy.utils.*;
import org.springframework.beans.BeanUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.util.CollectionUtils;
import org.springframework.util.ObjectUtils;

import javax.imageio.ImageIO;
import java.awt.*;
import java.io.File;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.math.BigDecimal;
import java.net.URLDecoder;
import java.util.*;
import java.util.List;
import java.util.stream.Collectors;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/22 20:05
 * @Description:
 */
@Service
public class DataMarkServiceImpl extends ServiceImpl<MarkInfoMapper, MarkInfoEntity> implements DataMarkService {

    @Autowired
    private DataSonMapper dataSonMapper;

    @Autowired
    private MarkInfoMapper markInfoMapper;

    @Autowired
    private DataSonLabelMapper dataSonLabelMapper;

    @Autowired
    private FileMapper fileMapper;

    @Autowired
    private DictDataService dictDataService;

    @Autowired
    private FileDownloadUtils fileDownloadUtils;

    @Autowired
    private MarkFileService markFileService;

    @Autowired
    private ManyAssignMapper manyAssignMapper;
    @Autowired
    private ManyMarkMapper manyMarkMapper;


    @Autowired
    private LabelMapper labelMapper;


    @Value("${file.accessAddress}")
    private String accessAddress;
    @Value("${file.versionFormat}")
    private String versionFormat;
    @Value("${upload.formalPath}")
    private String formalPath;

    @Autowired
    private JsonTransformer jsonTransformer;

    @Override
    public IPage<DataMarkResponse> getDataSetMarkList(Page<DataMarkResponse> pageParam, DataSonQueryRequest request) {
        IPage<DataMarkResponse> dataSetMarkList = dataSonMapper.getDataSetMarkList(pageParam, request);
        if (!CollectionUtils.isEmpty(dataSetMarkList.getRecords())) {
            for (DataMarkResponse markResponse : dataSetMarkList.getRecords()) {
                markResponse.setDataTypeName(dictDataService.getTreeLevelDict(markResponse.getDataTypeId()));
            }
        }
        return dataSetMarkList;
    }


    //----------------------------------------  保存标注信息  ----------------------------------------
    @Override
    public int addDataMarkInfo(MarkInfoEntity markInfoEntity) {
        DataSonEntity dataSonEntity = dataSonMapper.selectOne(
                new LambdaQueryWrapper<DataSonEntity>()
                        .eq(DataSonEntity::getSonId, markInfoEntity.getSonId())
        );

        MarkInfoEntity existingEntity = markInfoMapper.selectOne(
                new LambdaQueryWrapper<MarkInfoEntity>()
                        .eq(MarkInfoEntity::getSonId, markInfoEntity.getSonId())
                        .eq(MarkInfoEntity::getFileId, markInfoEntity.getFileId())
        );

        FileEntity fileEntity = fileMapper.selectById(markInfoEntity.getFileId());
        String markInfo = markInfoEntity.getMarkInfo();

        int result;
        if (ObjectUtils.isEmpty(existingEntity)) {
            // 执行新增
            addMarkInfo(dataSonEntity, markInfoEntity, fileEntity, markInfo);
            result = markInfoMapper.insert(markInfoEntity);
        } else {
            // 修改
            updateMarkInfo(dataSonEntity, existingEntity, markInfoEntity, fileEntity, markInfo);
            result = markInfoMapper.updateById(existingEntity);
        }

        // 记录标注进度
        recordMarkProgress(dataSonEntity, markInfoEntity);

        // 判断是否是多人标注
        if (markInfoEntity.getMarkUserId() != null) {
            updateManyMark(markInfoEntity.getMarkUserId());
        }
        return result;
    }

    // TODO 标注信息保存
    private void addMarkInfo(DataSonEntity dataSonEntity, MarkInfoEntity markInfoEntity, FileEntity fileEntity, String markInfo) {
        try {
            Image image = ImageIO.read(new File(fileEntity.getFdPath()));
            int width = image.getWidth(null);
            int height = image.getHeight(null);
            markInfoEntity.setWidth(width);
            markInfoEntity.setHeight(height);

            processMarkInfo(dataSonEntity, markInfoEntity, fileEntity, markInfo, width, height);
        } catch (IOException e) {
            throw new RuntimeException("标注信息保存失败！");
        }
    }

    // TODO  标注信息修改方法
    private void updateMarkInfo(DataSonEntity dataSonEntity, MarkInfoEntity existingEntity, MarkInfoEntity markInfoEntity, FileEntity fileEntity, String markInfo) {
        existingEntity.setMarkFileId(markInfoEntity.getMarkFileId());
        try {
            Image image = ImageIO.read(new File(fileEntity.getFdPath()));
            int width = image.getWidth(null);
            int height = image.getHeight(null);
            existingEntity.setWidth(width);
            existingEntity.setHeight(height);
            existingEntity.setOperateWidth(markInfoEntity.getOperateWidth());
            existingEntity.setOperateHeight(markInfoEntity.getOperateHeight());
            existingEntity.setIsInvalid(markInfoEntity.getIsInvalid());
            existingEntity.setMarkInfo(markInfoEntity.getMarkInfo());
            processMarkInfo(dataSonEntity, existingEntity, fileEntity, markInfo, width, height);
        } catch (IOException e) {
            throw new RuntimeException("标注信息保存失败！");
        }
    }

    // TODO  生成对应的数据
    //@Async
    public void processMarkInfo(DataSonEntity dataSonEntity, MarkInfoEntity entity, FileEntity fileEntity, String markInfo, int width, int height) {
        if (!StringUtils.isEmpty(markInfo) && !markInfo.equals("[]")) {
            List<WebRectangleShape> list = JSONUtil.toList(JSONUtil.parseArray(markInfo), WebRectangleShape.class);
            LabelmeImageData labelmeImageData = FormatConverter.convertWebDataToLabelme(list, fileEntity.getFdName(), width, height, entity);
            // TODO 替换里面的英文， 把里面的中文填写成对应的英文
            entity.setLabelMarkInfo(JSONUtil.toJsonStr(setLabelEnglish(labelmeImageData, entity)));


            fileDownloadUtils.writeFile(entity);

            // TODO 判断数据集标注类型是否是 图像分割 如果为图像分割

            if (dataSonEntity.getAnoType() == 1) {
                return;
            }

            String filePath = null;
            // TODO 生成出 第三方矩形框的数据
            filePath = formalPath + dataSonEntity.getFatherId() + versionFormat + dataSonEntity.getVersion()
                    + "/" + "rect/" + fileEntity.getFdName().split("\\.")[0] + ".json";
            //new JsonTransformer().transformJson(markInfo, filePath, dataSonEntity.getSonId());
            jsonTransformer.transformJson(entity, filePath, dataSonEntity.getSonId());

            // TODO 生成出 voc 格式的 xml
            filePath = formalPath + dataSonEntity.getFatherId() + versionFormat + dataSonEntity.getVersion()
                    + "/" + "xml/" + fileEntity.getFdName().split("\\.")[0] + ".xml";
            JsonToVocXml.convertToVocXml(fileEntity.getFdName(),
                    entity.getLabelMarkInfo(),
                    entity,
                    filePath);

        }
    }


    public LabelmeImageData setLabelEnglish(LabelmeImageData labelmeImageData, MarkInfoEntity markInfoEntity) {
        if (ObjectUtils.isEmpty(labelmeImageData)) {
            return labelmeImageData;

        }
        // 当前数据集的所有标签
        List<DataSonLabelEntity> list = dataSonLabelMapper.selectList(
                new LambdaQueryWrapper<DataSonLabelEntity>()
                        .eq(DataSonLabelEntity::getSonId, markInfoEntity.getSonId())
        );
        if (CollectionUtils.isEmpty(list)) {
            return labelmeImageData;
        }

        // 提取所有的 labelId
        List<Integer> labelIds = list.stream().map(DataSonLabelEntity::getLabelId).collect(Collectors.toList());
        // 批量查询标签信息
        List<LabelEntity> labelEntities = labelMapper.selectByIds(labelIds);

        // 使用 Map 存储标签名称映射
        Map<String, String> labelMap = new HashMap<>();
        for (LabelEntity labelEntity : labelEntities) {
            labelMap.put(labelEntity.getLabelName().trim(), labelEntity.getEnglishLabelName());
        }
        for (LabelmeShape shape : labelmeImageData.getShapes()) {
            String label = shape.getLabel();
            if (!StringUtils.isEmpty(label)) {
                String trimmedLabel = label.trim();
                if (labelMap.containsKey(trimmedLabel)) {
                    shape.setLabel(labelMap.get(trimmedLabel));
                }
            } else {
                shape.setLabel("null");
            }
        }
        return labelmeImageData;

    }

    // TODO  计算进度，并修改数据集
    private void recordMarkProgress(DataSonEntity dataSonEntity, MarkInfoEntity markInfoEntity) {


        if (!ObjectUtils.isEmpty(dataSonEntity)) {
            String[] fileIds = dataSonEntity.getFileIds().split(",");
            Integer count = markInfoMapper.selectCount(
                    new LambdaQueryWrapper<MarkInfoEntity>()
                            .eq(MarkInfoEntity::getSonId, dataSonEntity.getSonId())
            );
            int num = NumberUtil.div(count.toString(), Integer.toString(fileIds.length), 2).multiply(BigDecimal.valueOf(100)).intValue();
            dataSonEntity.setStatus(num + "% " + ("(" + count + "/" + fileIds.length + ")"));
            dataSonMapper.updateById(dataSonEntity);
        }
    }


    // --------------------------------------------------------------------------------------------------

    /**
     * 更改多人任务
     */
    /*private void updateManyMark(Integer markUserId) {
        ManyAssignEntity manyAssignEntity = manyAssignMapper.selectById(markUserId);
        if (ObjectUtils.isEmpty(manyAssignEntity)) {
            throw new RuntimeException("多人标注异常！");
        }
        // 文件总数量
        int fileNum = manyAssignEntity.getAssignFileIds().split(",").length;

        // TODO 未正常标注 （标注空白 或者 撤回）

        // 计算进度
        int yesMark = fileMapper.selectFileAndlabelYesMarkCount(manyAssignEntity.getAssignFileIds());
        manyAssignEntity.setProgress(MyProgressUtils.calculateCount(yesMark, fileNum) + "");
        manyAssignEntity.setYesMark(yesMark);
        manyAssignEntity.setNoMark(fileNum - yesMark);

        // 更改状态
        if (manyAssignEntity.getUserState() == 1) {
            manyAssignEntity.setUserState(2);
            // 任务表状态更改为 ： 标注中
            ManyMarkEntity manyMarkEntity = manyMarkMapper.selectById(manyAssignEntity.getManyMarkId());
            manyMarkEntity.setTaskState(3);
            manyMarkMapper.updateById(manyMarkEntity);
        }
        // 这里状态设置为已完成
        if (yesMark == fileNum) {
            manyAssignEntity.setUserState(3);
        }

        manyAssignMapper.updateById(manyAssignEntity);
    }
*/


    /**
     * 更改多人任务
     */
    private void updateManyMark(Integer markUserId) {
        ManyAssignEntity manyAssignEntity = manyAssignMapper.selectById(markUserId);
        if (ObjectUtils.isEmpty(manyAssignEntity)) {
            throw new RuntimeException("多人标注异常！");
        }

        // 1. 处理文件ID，获取有效文件总数（去重、去空）
        String assignFileIds = manyAssignEntity.getAssignFileIds();
        Set<String> validFileIds = new HashSet<>();
        if (StringUtils.isNotBlank(assignFileIds)) {
            String[] fileIdArray = assignFileIds.split(",");
            for (String fileId : fileIdArray) {
                if (StringUtils.isNotBlank(fileId)) {
                    validFileIds.add(fileId.trim());
                }
            }
        }
        int fileNum = validFileIds.size();

        // 2. 统计已标注数量（确认返回值非null，直接使用int接收）
        String validFileIdsStr = StringUtils.join(validFileIds, ",");
        int yesMark = fileMapper.selectFileAndlabelYesMarkCount(validFileIdsStr) + fileMapper.selectFileInvalidDataCount(validFileIdsStr); // 直接用int接收

        // 3. 计算未标注数量（确保非负）
        int noMark = Math.max(0, fileNum - yesMark);

        // 4. 计算进度（处理边界情况）
        int progress = 0;
        if (fileNum > 0) {
            int actualYesMark = Math.min(yesMark, fileNum); // 防止已标注数超过总文件数
            progress = MyProgressUtils.calculateCount(actualYesMark, fileNum);
            progress = Math.min(100, Math.max(0, progress)); // 限制进度范围
        }

        // 5. 更新进度和数量信息
        manyAssignEntity.setProgress(String.valueOf(progress));
        manyAssignEntity.setYesMark(yesMark);
        manyAssignEntity.setNoMark(noMark);

        manyAssignMapper.updateById(manyAssignEntity);
    }









    /*@Override
    public int addDataMarkInfo(MarkInfoEntity markInfoEntity) {
        // 判断之前是否标记过，如果标记过则进行修改
        MarkInfoEntity entity = markInfoMapper.selectOne(
                new LambdaQueryWrapper<MarkInfoEntity>()
                        .eq(MarkInfoEntity::getSonId, markInfoEntity.getSonId())
                        .eq(MarkInfoEntity::getFileId, markInfoEntity.getFileId())
        );
        int result;
        FileEntity fileEntity = fileMapper.selectById(markInfoEntity.getFileId());
        String markInfo = markInfoEntity.getMarkInfo();
        Image image = null;
        if (ObjectUtils.isEmpty(entity)) {
            // 执行新增
            try {
                image = ImageIO.read(new File(fileEntity.getFdPath()));
                int width = image.getWidth(null);
                int height = image.getHeight(null);
                markInfoEntity.setWidth(width);
                markInfoEntity.setHeight(height);
                if (markInfoEntity.getIsInvalid() == 1) {
                    if (!StringUtils.isEmpty(markInfoEntity.getMarkInfo())) {
                        List<WebRectangleShape> list = JSONUtil.toList(JSONUtil.parseArray(markInfo), WebRectangleShape.class);
                        List<WebRectangleShape> point = list.stream().filter(item -> {
                            return item.getType().equalsIgnoreCase("point");
                        }).collect(Collectors.toList());
                        List<WebRectangleShape> mark = list.stream().filter(item -> {
                            return !item.getType().equalsIgnoreCase("point");
                        }).collect(Collectors.toList());
                        LabelmeImageData labelmeImageData = FormatConverter.convertWebDataToLabelme(mark, fileEntity.getFdName(), width, height, markInfoEntity);
                        markInfoEntity.setLabelMarkInfo(JSONUtil.toJsonStr(labelmeImageData));
                        // 生成json文件
                        fileDownloadUtils.writeFile(markInfoEntity);
                    }
                } else {
                    markInfoEntity.setMarkFileId(0);
                    markInfoEntity.setLabels("");
                    markInfoEntity.setMarkInfo("");
                    markInfoEntity.setLabelMarkInfo("");
                }

            } catch (IOException e) {
                System.out.println("label info失败");
            }
            result = markInfoMapper.insert(markInfoEntity);
        } else { //修改
            entity.setMarkFileId(markInfoEntity.getMarkFileId());
            try {
                image = ImageIO.read(new File(fileEntity.getFdPath()));
                int width = image.getWidth(null);
                int height = image.getHeight(null);
                // 判断（前端传过来的）标签 不能为空字符或者null 或者 空数组
                entity.setWidth(width);
                entity.setHeight(height);
                entity.setOperateWidth(markInfoEntity.getOperateWidth());
                entity.setOperateHeight(markInfoEntity.getOperateHeight());
                entity.setIsInvalid(markInfoEntity.getIsInvalid());
                if (markInfoEntity.getIsInvalid() == 1) {
                    if (!StringUtils.isEmpty(markInfo) && !markInfo.equals("[]")) {
                        entity.setLabels(markInfoEntity.getLabels());
                        entity.setMarkInfo(markInfoEntity.getMarkInfo());
                        List<WebRectangleShape> list = JSONUtil.toList(JSONUtil.parseArray(markInfo), WebRectangleShape.class);
                        LabelmeImageData labelmeImageData = FormatConverter.convertWebDataToLabelme(list, fileEntity.getFdName(), width, height, entity);
                        entity.setLabelMarkInfo(JSONUtil.toJsonStr(labelmeImageData));
                        fileDownloadUtils.writeFile(entity);
                    }
                } else {
                    entity.setMarkFileId(0);
                    entity.setLabels("");
                    entity.setLabelMarkInfo("");
                    entity.setMarkInfo("");
                }

            } catch (IOException e) {
                System.out.println("label info失败");
            }

            result = markInfoMapper.updateById(entity);
        }


        // 记录标注进度
        DataSonEntity dataSonEntity = dataSonMapper.selectOne(
                new LambdaQueryWrapper<DataSonEntity>()
                        .eq(DataSonEntity::getSonId, markInfoEntity.getSonId())
        );

        // 计算进度
        if (!ObjectUtils.isEmpty(dataSonEntity)) {
            String[] fileIds = dataSonEntity.getFileIds().split(",");
            Integer count = markInfoMapper.selectCount(
                    new LambdaQueryWrapper<MarkInfoEntity>()
                            .eq(MarkInfoEntity::getSonId, dataSonEntity.getSonId())
            );
            int num = NumberUtil.div(count.toString(), Integer.toString(fileIds.length), 2).multiply(BigDecimal.valueOf(100)).intValue();
            dataSonEntity.setStatus(num + "% " + ("(" + count + "/" + fileIds.length + ")"));
            dataSonMapper.updateById(dataSonEntity);
        }

        // 判断是否是多人标注
        if (markInfoEntity.getMarkUserId() != null) {
            updateManyMark(markInfoEntity.getMarkUserId());
        }


        return result;
    }*/


    @Override
    public List<DataDetailsResponse> getDataDetails(String sonId, Integer state) {
        try {
            DataSonEntity dataSonEntity = dataSonMapper.selectOne(
                    new LambdaQueryWrapper<DataSonEntity>()
                            .eq(DataSonEntity::getSonId, sonId)
            );
            if (ObjectUtils.isEmpty(dataSonEntity)) {
                throw new RuntimeException("后端异常，数据集不存在！");
            }

            if (StringUtils.isEmpty(dataSonEntity.getFileIds())) {
                throw new RuntimeException("后端异常，文件id为空！");
            }


            List<DataDetailsResponse> list = null;

            switch (state) {
                case 0:      // 0.全部数据
                    list = fileMapper.selectFileAndlabelNoPage(dataSonEntity.getFileIds());

                    if (!CollectionUtils.isEmpty(list)) {
                        // 进行遍历
                        putPrice(dataSonEntity, list);
                    }
                    break;

                case 1:    // 1.有标注信息
                    list = fileMapper.selectFileAndlabelYesMarkNoPage(dataSonEntity.getFileIds());
                    if (!CollectionUtils.isEmpty(list)) {
                        // 进行遍历
                        putPrice(dataSonEntity, list);
                    }
                    break;

                case 2:  // 无标注信息
                    list = fileMapper.selectFileAndlabelNoMarkNoPage(dataSonEntity.getFileIds());
                    if (!CollectionUtils.isEmpty(list)) {
                        // 进行遍历
                        putPrice(dataSonEntity, list);
                    }
                    break;
            }
            return list;

        } catch (RuntimeException e) {
            throw new RuntimeException(e);
        }
    }


    @Override
    public void setMarkFileJsonWrite(String outSonId) {
        try {
            DataSonEntity dataSonEntity = dataSonMapper.selectOne(
                    new LambdaQueryWrapper<DataSonEntity>()
                            .eq(DataSonEntity::getSonId, outSonId)
            );
            if (ObjectUtils.isEmpty(dataSonEntity)) {
                throw new RuntimeException("数据集不存在!");
            }
            if (StringUtils.isEmpty(dataSonEntity.getFileIds())) {
                throw new RuntimeException("文件id为空!");
            }


            // 定义临时文件集合
            List<FileEntity> fileEntityList = new ArrayList<>();

            // 分割文件id
            for (String fileIdStr : dataSonEntity.getFileIds().split(",")) {
                Integer fileId = Integer.parseInt(fileIdStr);

                FileEntity fileEntity = fileMapper.selectById(fileId);
                if (ObjectUtils.isEmpty(fileEntity)) {
                    continue;
                }

                FileEntity fileEntity1 = new FileEntity();
                BeanUtils.copyProperties(fileEntity, fileEntity1);
                fileEntity1.setFdName(fileEntity.getFdName().split("\\.")[0]);
                fileEntityList.add(fileEntity1);
               /* // 取出文件名前缀（不加.类型）
                String filePrefixName = fileEntity.getFdName().split("\\.")[0];
                fileEntity1.setId(fileEntity.getId());
                fileEntity1.setFdName(filePrefixName);
                fileEntity1.setFdSuffix(fileEntity.getFdSuffix());
                fileEntity1.setFdPath(fileEntity.getFdPath());
                if (!fileEntity.getFdSuffix().equals(".json")) {
                    fileEntity1.setWidth(fileEntity.getWidth());
                    fileEntity1.setHeight(fileEntity.getHeight());
                    fileEntity1.setOperateWidth(fileEntity.getOperateWidth());
                    fileEntity1.setOperateHeight(fileEntity.getOperateHeight());
                }
*/
            }
            // 使用stream流根据名字进行分组
            Map<String, List<FileEntity>> data = fileEntityList.stream().collect(
                    Collectors.groupingBy(
                            FileEntity::getFdName
                    )
            );

            // 记录文件id
            StringBuilder sb = new StringBuilder();

            // 遍历map集合
            for (String key : data.keySet()) {
                // 根据key获取value
                List<FileEntity> valueFileEntity = data.get(key);
                for (FileEntity fileEntity : valueFileEntity) {
                    // 判断是否是图片
                    if (ImageUtils.isImage(fileEntity.getFdPath())) {
                        MarkInfoEntity markInfoEntity = new MarkInfoEntity();
                        markInfoEntity.setFileId(fileEntity.getId());
                        markInfoEntity.setSonId(dataSonEntity.getSonId());
                        markInfoEntity.setMarkFileId(fileEntity.getId());
                        sb.append(fileEntity.getId()).append(",");
                        markInfoEntity.setCreateTime(new Date());
                        markInfoEntity.setWidth(fileEntity.getWidth());
                        markInfoEntity.setHeight(fileEntity.getHeight());
                        markInfoEntity.setOperateWidth(fileEntity.getOperateWidth());
                        markInfoEntity.setOperateHeight(fileEntity.getOperateHeight());
                        markInfoMapper.insert(markInfoEntity);
                    }
                }
                MarkInfoEntity markEntity = null;
                for (FileEntity fileEntity : valueFileEntity) {


                    if (ImageUtils.isImage(fileEntity.getFdPath())) {
                        markEntity = markInfoMapper.selectOne(
                                new LambdaQueryWrapper<MarkInfoEntity>()
                                        .eq(MarkInfoEntity::getSonId, outSonId)
                                        .eq(MarkInfoEntity::getFileId, fileEntity.getId())
                        );
                    }


                    if (fileEntity.getFdSuffix().equals(".json")) {
                        // 修改
                        if (ObjectUtils.isEmpty(markEntity)) {
                            continue;
                        }

                        String jsonStr = FileToStringUtils.readTextFile(fileEntity.getFdPath());
                        markEntity.setLabelMarkInfo(jsonStr);
                        LabelmeImageData bean = JSONUtil.toBean(jsonStr, LabelmeImageData.class);
                        System.out.println(bean);
                        List<WebRectangleShape> webRectangleShapes = FormatConverter.convertLabelmeDataToWeb(bean, markEntity);
                        markEntity.setMarkInfo(JSONUtil.toJsonStr(webRectangleShapes));
                        markInfoMapper.updateById(markEntity);

                        // 生成json文件
                        fileDownloadUtils.writeFile(markEntity);
                    }
                }
            }

            // 最后修改数据集的文件id数据
            String fileIds = sb.toString().replaceAll("^,|,$", "");
            System.out.println("文件id：" + fileIds);
            dataSonEntity.setFileIds(fileIds);
            dataSonEntity.setStatus("0% (0/" + fileIds.split(",").length + ")");
            dataSonMapper.updateById(dataSonEntity);

        } catch (Exception e) {
            throw new RuntimeException(e);
        }

    }

    @Override
    public void setMarkFileJsonAndMarkFileWrite(String sonId, String fileId, String jsonId, String markFileId) {
        try {
            // 获取数据集信息
            DataSonEntity dataSonEntity = dataSonMapper.selectOne(
                    new LambdaQueryWrapper<DataSonEntity>()
                            .eq(DataSonEntity::getSonId, sonId)
            );
            if (ObjectUtils.isEmpty(dataSonEntity)) {
                throw new RuntimeException("数据集不存在！");
            }
            // 获取原始图信息
            FileEntity sourceFileEntity = fileMapper.selectById(Integer.parseInt(fileId));
            if (ObjectUtils.isEmpty(sourceFileEntity)) {
                throw new RuntimeException("原始图数据不存在！");
            }
            // 获取json图信息
            FileEntity jsonFileEntity = fileMapper.selectById(Integer.parseInt(jsonId));
            if (ObjectUtils.isEmpty(jsonFileEntity)) {
                throw new RuntimeException("json文件数据不存在！");
            }
            // 获取标注图文件信息
            FileEntity markFileEntity = fileMapper.selectById(Integer.parseInt(markFileId));
            if (ObjectUtils.isEmpty(markFileEntity)) {
                throw new RuntimeException("标注文件不存在！");
            }
            MarkInfoEntity oldMarkInfo = markInfoMapper.selectOne(
                    new LambdaQueryWrapper<MarkInfoEntity>()
                            .eq(MarkInfoEntity::getSonId, dataSonEntity.getSonId())
                            .eq(MarkInfoEntity::getFileId, sourceFileEntity.getId())
            );

            // 解析json信息
            MarkInfoEntity markInfoEntity = new MarkInfoEntity();
            markInfoEntity.setSonId(dataSonEntity.getSonId());
            markInfoEntity.setFileId(sourceFileEntity.getId());
            // 获取文件的宽和高
            if (ImageUtils.isImage(markFileEntity.getFdPath())) {
                int[] ints = ImageUtils.getImageDimensions(markFileEntity.getFdPath());
                markInfoEntity.setWidth(ints[0]);
                markFileEntity.setHeight(ints[1]);
                markInfoEntity.setOperateWidth(ints[0]);
                markInfoEntity.setOperateHeight(ints[1]);
            } else {
                throw new RuntimeException("标注文件类型不是图片！");
            }
            String jsonStr = FileToStringUtils.readTextFile(jsonFileEntity.getFdPath());
            // 前端需要的json信息
            markInfoEntity.setLabelMarkInfo(jsonStr);
            LabelmeImageData bean = JSONUtil.toBean(jsonStr, LabelmeImageData.class);
            List<WebRectangleShape> webRectangleShapes = FormatConverter.convertLabelmeDataToWeb(bean, markInfoEntity);

            // 解析标签
            String labelIds = setDataSetLabel(webRectangleShapes, dataSonEntity.getSonId());
            if (!StringUtils.isEmpty(labelIds)) {
                markInfoEntity.setLabels(labelIds);
            }

            // 算法需要的json信息
            markInfoEntity.setMarkInfo(JSONUtil.toJsonStr(webRectangleShapes));

            // 上传图片到这个数据集
            Integer newFileId = markFileService.addMarkFile(sonId, markFileEntity.getId());
            markInfoEntity.setMarkFileId(newFileId);


            if (ObjectUtils.isEmpty(oldMarkInfo)) {
                markInfoMapper.insert(markInfoEntity);
                // 生成json文件
                fileDownloadUtils.writeFile(markInfoEntity);
            } else {
                BeanUtil.copyProperties(markInfoEntity, oldMarkInfo, CopyOptions.create().setIgnoreNullValue(true));
//                BeanUtils.copyProperties(markInfoEntity, oldMarkInfo);
                markInfoMapper.updateById(oldMarkInfo);
                // 生成json文件
                fileDownloadUtils.writeFile(oldMarkInfo);
            }


            // 计算数据集进度
            String[] files = dataSonEntity.getFileIds().split(",");
            Integer count = markInfoMapper.selectCount(
                    new LambdaQueryWrapper<MarkInfoEntity>()
                            .eq(MarkInfoEntity::getSonId, dataSonEntity.getSonId())
            );
            int num = NumberUtil.div(count.toString(), Integer.toString(files.length), 2).multiply(BigDecimal.valueOf(100)).intValue();
            dataSonEntity.setStatus(num + "% " + ("(" + count + "/" + files.length + ")"));
            dataSonMapper.updateById(dataSonEntity);


        } catch (Exception e) {
            throw new RuntimeException(e);
        }

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

            // 查询是否存在
            if (dataSonLabelMapper.selectBySonIdAndLabelNameCount(sonId, webRectangleShape.getProps().getName()) > 0) {
                continue;
            }

            // 实现新增
            LabelEntity labelEntity = new LabelEntity();
            labelEntity.setLabelName(webRectangleShape.getProps().getName());
            labelEntity.setLabelColor("#D91515");
            labelEntity.setLabelGroupId(0);
            labelEntity.setCreateTime(new Date());
            labelEntity.setLabelGroupId(0);
            labelMapper.insert(labelEntity);

            // 关联数据集 AND 标签表
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


    public void putPrice(DataSonEntity dataSonEntity, List<DataDetailsResponse> list) {
        String prefixToRemove = formalPath;
        try {
            // 进行遍历
            for (DataDetailsResponse response : list) {
                response.setSonId(dataSonEntity.getSonId());
                response.setVersion(dataSonEntity.getVersion());
                // 原始图
                FileEntity fileEntity = fileMapper.selectById(response.getFileId());

                response.setImgPath(URLUtils.encodeURL(URLDecoder.decode(accessAddress + fileEntity.getFdPath().replaceFirst("^" + prefixToRemove, ""), "UTF-8")));
                response.setPreviewImgPath(URLUtils.encodeURL(URLDecoder.decode(accessAddress + fileEntity.getHttpFilePath().replaceFirst("^" + prefixToRemove, ""), "UTF-8")));

                /*response.setImgPath(
                        URLUtils.encodeURL(
                                accessAddress + "/" + dataSonEntity.getFatherId() + "/" + "v" + dataSonEntity.getVersion()
                                        + "/" + "source" + "/" + fileEntity.getFdName()
                        )
                );*/

            }
        } catch (UnsupportedEncodingException e) {
            throw new RuntimeException(e);
        }
    }


    private static Map<Integer, Integer> calculationNum(int[] arr) {
        Map<Integer, Integer> map = new HashMap<Integer, Integer>();
        for (int i = 0; i < arr.length; i++) {
            if (map.get(arr[i]) != null) {
                map.put(arr[i], map.get(arr[i]) + 1);
            } else {
                map.put(arr[i], 1);
            }
        }
        return map;
    }


    // 实现文件上传记录
    private Integer addUploadEntity(FileEntity fileEntity) {
        fileMapper.insert(fileEntity);
        return fileEntity.getId();
    }


}


