package com.qczy.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.api.R;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;

import com.qczy.common.generate.GeneratePdfForm;
import com.qczy.common.word.GetDynamicData;
import com.qczy.mapper.*;
import com.qczy.model.entity.*;
import com.qczy.model.request.DataSonEntityRequest;
import com.qczy.model.request.ModelApplyForRequest;
import com.qczy.model.request.ModelApplyForRequestParam;
import com.qczy.model.request.ModelBackFillRequest;
import com.qczy.model.response.ModelApplyForListResponse;
import com.qczy.service.DataSonService;
import com.qczy.service.ModelApplyForService;
import com.qczy.service.ModelCodeService;
import com.qczy.utils.CurrentLoginUserUtils;
import com.qczy.utils.StringUtils;
import org.apache.commons.lang3.RandomStringUtils;
import org.jpedal.parser.shape.F;
import org.springframework.beans.BeanUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.util.CollectionUtils;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.File;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.stream.Collectors;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/5/16 11:24
 * @Description:
 */
@Service
public class ModelApplyForServiceImpl implements ModelApplyForService {

    @Autowired
    private ModelBaseMapper modelBaseMapper;

    @Autowired
    private ModelConfigureMapper modelConfigureMapper;

    @Autowired
    private CurrentLoginUserUtils currentLoginUserUtils;
    @Autowired
    private TempFileMapper tempFileMapper;

    @Autowired
    private GeneratePdfForm generatePdfForm;

    @Autowired
    private DataSonService dataSonService;

    @Autowired
    private DictTypeMapper dictTypeMapper;
    @Autowired
    private DictDataMapper dictDataMapper;

    @Autowired
    private ModelCodeService modelCodeService;

    @Autowired
    private ModelCodeMapper modelCodeMapper;

    @Autowired
    private DataSonMapper dataSonMapper;

    @Autowired
    private DataFatherMapper dataFatherMapper;

    @Value("${upload.address}")
    private String uploadAddress;


    @Override
    @Transactional
    public ModelApplyForListResponse addModel(ModelApplyForRequest request) {
        ModelBaseEntity modelBaseEntity = new ModelBaseEntity();
        BeanUtils.copyProperties(request, modelBaseEntity);
        boolean control = true;
        // 判断申请单号是否重复，如果重复，则接着生成
        do {
            // 申请单号
            String dateStr = new SimpleDateFormat("yyyyMMdd").format(new Date());
            String applyNo = dateStr + ((int) (Math.random() * 900) + 100);
            if (!isApplyNo(applyNo)) {
                modelBaseEntity.setApplyForNum(applyNo); //申请单号
                control = false;
            }
        } while (control);
        modelBaseEntity.setApplyForDate(new Date()); // 申请时间
        modelBaseEntity.setUserId(currentLoginUserUtils.getCurrentLoginUserId()); // 申请人
        modelBaseEntity.setApplyForStatus(1); //申请状态
        modelBaseMapper.insert(modelBaseEntity);

        // 新增配置表
        ModelConfigureEntity modelConfigureEntity = new ModelConfigureEntity();
        BeanUtils.copyProperties(request, modelConfigureEntity);
        if (request.getFileId() != null) {
            // 新增数据集
            String sonId = addDataSon(request.getFileId());
            modelConfigureEntity.setSonId(sonId);
        }
        modelConfigureEntity.setModelBaseId(modelBaseEntity.getId()); // 配置基础信息id

        int result = modelConfigureMapper.insert(modelConfigureEntity);
        // 解析算法编码
        if (!StringUtils.isEmpty(request.getModelAlgorithmCode())) {
            //System.out.println("编码xlsx路径= " + request.getModelAlgorithmCode());
            modelCodeService.analysisXlsxCode(request.getModelAlgorithmCode(), modelBaseEntity.getId());
        }
        // 组织结构返回给前端
        ModelApplyForListResponse response = new ModelApplyForListResponse();
        response.setId(modelBaseEntity.getId());
        response.setApplyForStatus(modelBaseEntity.getApplyForStatus());
        response.setApplyForType(modelBaseEntity.getApplyForType());
        return response;
    }


    // 查询申请单号是否存在，如果存在，则换一个
    public boolean isApplyNo(String applyNo) {
        return modelBaseMapper.selectCount(
                new LambdaQueryWrapper<ModelBaseEntity>().eq(ModelBaseEntity::getApplyForNum, applyNo)
        ) > 0;
    }


    // 新增数据集
    public String addDataSon(String fileId) {
        DataSonEntityRequest dataSonEntityRequest = new DataSonEntityRequest();

        // 随机命名：前缀 + 6位随机字母数字 + 日期
        String randomStr = RandomStringUtils.randomAlphanumeric(6); // 生成6位随机字母数字组合
        String prefixFileName = "第三方数据集-" + randomStr + "-" + new SimpleDateFormat("yyyyMMdd").format(new Date());
        dataSonEntityRequest.setFileIds(fileId);
        /**
         *  动态获取数据集字典id
         */
        dataSonEntityRequest.setDataTypeId(getDataSonDictId());
        dataSonEntityRequest.setGroupName(prefixFileName);
        dataSonEntityRequest.setMarkStatus(1);
        dataSonEntityRequest.setImportMode(2);
        dataSonEntityRequest.setVersion(1);
        dataSonEntityRequest.setAnoType(0);
        dataSonEntityRequest.setIsMany(0);
        dataSonService.insertDataSet(dataSonEntityRequest);
        return dataSonEntityRequest.getSonId();
    }

    private int getDataSonDictId() {
        // 首先先查询是否存在
        List<DictDataEntity> dictDataEntities = dictDataMapper.selectList(
                new LambdaQueryWrapper<DictDataEntity>()
                        .eq(DictDataEntity::getTypeId, 6)
                        .eq(DictDataEntity::getParentId, 0)
                        .eq(DictDataEntity::getDictLabel, "第三方数据集")
        );

        if (!dictDataEntities.isEmpty()) {
            // 说明存在，直接返回
            return dictDataEntities.get(0).getId();
        } else {
            // 不存在，进行新增
            DictDataEntity dictDataEntity = new DictDataEntity();
            dictDataEntity.setDictLabel("第三方数据集");
            dictDataEntity.setParentId(0);
            dictDataEntity.setTypeId(6);
            dictDataEntity.setDictSort(999);
            dictDataEntity.setCreateTime(new Date());
            dictDataMapper.insert(dictDataEntity);
            return dictDataEntity.getId();
        }
    }


    @Override
    public int editModel(ModelApplyForRequest request) {
        ModelBaseEntity modelBaseEntity = modelBaseMapper.selectById(request.getId());
        BeanUtils.copyProperties(request, modelBaseEntity);
        modelBaseMapper.updateById(modelBaseEntity);


        ModelConfigureEntity modelConfigureEntity = modelConfigureMapper.selectOne(
                new LambdaQueryWrapper<ModelConfigureEntity>().eq(ModelConfigureEntity::getModelBaseId, modelBaseEntity.getId())
        );
        request.setId(modelConfigureEntity.getId());
        BeanUtils.copyProperties(request, modelConfigureEntity);

        if (request.getFileId() != null) {
            // 新增数据集
            String sonId = addDataSon(request.getFileId());
            modelConfigureEntity.setSonId(sonId);
        }

        // 如果修改了编码，则进行删除后再次新增
        if (!StringUtils.isEmpty(request.getModelAlgorithmCode())) {
            // 首先先删除
            List<ModelCodeEntity> list = modelCodeMapper.selectList(
                    new LambdaQueryWrapper<ModelCodeEntity>()
                            .eq(ModelCodeEntity::getModelBaseId, modelBaseEntity.getId())
            );
            if (!CollectionUtils.isEmpty(list)) {
                List<Integer> ids = list.stream().map(ModelCodeEntity::getId).collect(Collectors.toList());
                modelCodeMapper.deleteBatchIds(ids);
            }
            // 然后解析新增
            modelCodeService.analysisXlsxCode(request.getModelAlgorithmCode(), modelBaseEntity.getId());
        }
        return modelConfigureMapper.updateById(modelConfigureEntity);
    }


    @Override
    public IPage<ModelApplyForListResponse> list(Page<ModelApplyForListResponse> pageParam, ModelApplyForRequestParam requestParam) {
        // 获取当时是否是领导登录
        boolean isAdminLogin = currentLoginUserUtils.getCurrentLoginUserModelIds();
        // 当前登录用户
        Integer userId = currentLoginUserUtils.getCurrentLoginUserId();

        return modelBaseMapper.list(pageParam, requestParam, isAdminLogin, userId);
    }

    @Override
    public IPage<ModelApplyForListResponse> approveList(Page<ModelApplyForListResponse> pageParam, ModelApplyForRequestParam requestParam) {
        return modelBaseMapper.approveList(pageParam, requestParam);
    }

    @Autowired
    private GetDynamicData getDynamicData;

    @Override
    public void generatePad(Integer modelId, HttpServletRequest request, HttpServletResponse response) {
        //generatePdfForm.downloadPdf(modelId, request, response);
        getDynamicData.downloadWord(modelId, request, response);
    }

    @Override
    public ModelApplyForRequest modelDetails(Integer modelId) {
        ModelBaseEntity modelBaseEntity = modelBaseMapper.selectById(modelId);
        if (modelBaseEntity == null) {
            return null;
        }
        ModelConfigureEntity modelConfigureEntity =
                modelConfigureMapper.selectOne(new LambdaQueryWrapper<ModelConfigureEntity>().eq(ModelConfigureEntity::getModelBaseId, modelBaseEntity.getId()));
        if (modelConfigureEntity == null) {
            return null;
        }

        ModelApplyForRequest modelApplyForRequest = new ModelApplyForRequest();

        BeanUtils.copyProperties(modelBaseEntity, modelApplyForRequest);
        BeanUtils.copyProperties(modelConfigureEntity, modelApplyForRequest);

        // 回显文件名称
      /*  if (!StringUtils.isEmpty(modelConfigureEntity.getModelInterfaceDesc())) {
            File file = new File(modelConfigureEntity.getModelInterfaceDesc());
            if (file.exists())
                modelApplyForRequest.setModelInterfaceDescFileName(file.getName());
        }
        if (!StringUtils.isEmpty(modelConfigureEntity.getModelCase())) {
            File file = new File(modelConfigureEntity.getModelCase());
            if (file.exists())
                modelApplyForRequest.setModelCaseFileName(file.getName());
        }
        if (!StringUtils.isEmpty(modelConfigureEntity.getModelAlgorithmCode())) {
            File file = new File(modelConfigureEntity.getModelAlgorithmCode());
            if (file.exists())
                modelApplyForRequest.setModelAlgorithmCodeFileName(file.getName());
        }*/

        if (!StringUtils.isEmpty(modelApplyForRequest.getModelInterfaceDesc()))
            modelApplyForRequest.setModelInterfaceDesc(uploadAddress + modelApplyForRequest.getModelInterfaceDesc());

        if (!StringUtils.isEmpty(modelApplyForRequest.getModelCase()))
            modelApplyForRequest.setModelCase(uploadAddress + modelApplyForRequest.getModelCase());


        if (!StringUtils.isEmpty(modelApplyForRequest.getModelTrainCode()))
            modelApplyForRequest.setModelTrainCode(uploadAddress + modelApplyForRequest.getModelTrainCode());


        if (!StringUtils.isEmpty(modelApplyForRequest.getModelAlgorithmCode()))
            modelApplyForRequest.setModelAlgorithmCode(uploadAddress + modelApplyForRequest.getModelAlgorithmCode());


        if (!StringUtils.isEmpty(modelApplyForRequest.getTestCase()))
            modelApplyForRequest.setTestCase(uploadAddress + modelApplyForRequest.getTestCase());

        if (!StringUtils.isEmpty(modelConfigureEntity.getSonId())) {
            DataSonEntity dataSonEntity = dataSonMapper.getDataSonBySonId(modelConfigureEntity.getSonId());
            if (dataSonEntity != null) {
                DataFatherEntity dataFatherEntity = dataFatherMapper.selectOne(
                        new LambdaQueryWrapper<DataFatherEntity>().eq(DataFatherEntity::getGroupId, dataSonEntity.getFatherId())
                );
                if (dataFatherEntity != null) {
                    modelApplyForRequest.setGroupNameAndVersion(dataFatherEntity.getGroupName() + "/v" + dataSonEntity.getVersion());
                }

            }
        }


        return modelApplyForRequest;
    }

    @Override
    public int modelBackFill(ModelBackFillRequest request) {
        ModelBaseEntity modelBaseEntity = modelBaseMapper.selectById(request.getId());
        if (modelBaseEntity == null) {
            return 0;
        }
        // 回填
        modelBaseEntity.setApplyForPdf(request.getFilePath());
        return modelBaseMapper.updateById(modelBaseEntity);
    }

    @Override
    public int submitApprove(ModelBaseEntity modelBaseEntity) {
        modelBaseEntity.setApplyForStatus(2);
        modelBaseEntity.setApproveStatus(1);
        return modelBaseMapper.updateById(modelBaseEntity);
    }


    public static void main(String[] args) {
        int i = (int) (Math.random() * 900) + 100;
        System.out.println(i);
    }
}
