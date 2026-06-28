// Copyright 1998-2017 Epic Games, Inc. All Rights Reserved.

#include "Blocks.h"

#include "Cesium3DTileset.h"
#include "CesiumGeoreference.h"
#include "Engine/StaticMeshActor.h"
#include "Engine/World.h"
#include "EngineUtils.h"
#include "GameFramework/Actor.h"
#include "GameFramework/WorldSettings.h"
#include "HAL/PlatformMisc.h"
#include "Modules/ModuleManager.h"

namespace {
const FName DroneRsiCesiumTag(TEXT("DroneRSI_CesiumSF"));
const FName DroneRsiGeorefTag(TEXT("DroneRSI_SF_Georeference"));

double ReadEnvDouble(const TCHAR* Name, double DefaultValue) {
  const FString Value = FPlatformMisc::GetEnvironmentVariable(Name);
  if (Value.IsEmpty()) {
    return DefaultValue;
  }
  const double Parsed = FCString::Atod(*Value);
  return FMath::IsFinite(Parsed) ? Parsed : DefaultValue;
}

int64 ReadEnvInt64(const TCHAR* Name, int64 DefaultValue) {
  const FString Value = FPlatformMisc::GetEnvironmentVariable(Name);
  if (Value.IsEmpty()) {
    return DefaultValue;
  }
  const int64 Parsed = FCString::Atoi64(*Value);
  return Parsed > 0 ? Parsed : DefaultValue;
}
} // namespace

class FBlocksModule : public FDefaultGameModuleImpl {
public:
  virtual void StartupModule() override {
    PostWorldInitHandle = FWorldDelegates::OnPostWorldInitialization.AddRaw(
        this, &FBlocksModule::HandlePostWorldInitialization);
  }

  virtual void ShutdownModule() override {
    if (PostWorldInitHandle.IsValid()) {
      FWorldDelegates::OnPostWorldInitialization.Remove(PostWorldInitHandle);
      PostWorldInitHandle.Reset();
    }
  }

private:
  void HandlePostWorldInitialization(
      UWorld* World,
      const UWorld::InitializationValues IVS) {
    if (!World) {
      return;
    }

    const EWorldType::Type WorldType = World->WorldType;
    if (WorldType != EWorldType::Game && WorldType != EWorldType::PIE) {
      return;
    }

    if (AWorldSettings* WorldSettings = World->GetWorldSettings()) {
      WorldSettings->bEnableWorldBoundsChecks = false;
    }

    for (TActorIterator<ACesium3DTileset> It(World); It; ++It) {
      if (It->Tags.Contains(DroneRsiCesiumTag)) {
        return;
      }
    }

    const FString IonToken =
        FPlatformMisc::GetEnvironmentVariable(TEXT("CESIUM_ION_TOKEN"));
    if (IonToken.IsEmpty()) {
      UE_LOG(
          LogTemp,
          Warning,
          TEXT("DroneRSI: CESIUM_ION_TOKEN is not set; SF Cesium tileset was not spawned."));
      return;
    }

    const double OriginLat =
        ReadEnvDouble(TEXT("SF_ORIGIN_LAT"), 37.7749);
    const double OriginLon =
        ReadEnvDouble(TEXT("SF_ORIGIN_LON"), -122.4194);
    const double OriginHeight =
        ReadEnvDouble(TEXT("SF_ORIGIN_ALT"), 0.0);
    const int64 IonAssetId =
        ReadEnvInt64(TEXT("CESIUM_ION_ASSET_ID"), 2275207);

    ACesiumGeoreference* Georeference = World->SpawnActor<ACesiumGeoreference>(
        ACesiumGeoreference::StaticClass(),
        FTransform::Identity);
    if (!Georeference) {
      UE_LOG(LogTemp, Error, TEXT("DroneRSI: failed to spawn Cesium georeference."));
      return;
    }

    Georeference->Tags.Add(DroneRsiGeorefTag);
    Georeference->SetOriginLongitudeLatitudeHeight(
        FVector(OriginLon, OriginLat, OriginHeight));
    Georeference->SetScale(100.0);

    ACesium3DTileset* Tileset =
        World->SpawnActorDeferred<ACesium3DTileset>(
            ACesium3DTileset::StaticClass(),
            FTransform::Identity);
    if (!Tileset) {
      UE_LOG(LogTemp, Error, TEXT("DroneRSI: failed to spawn Cesium SF tileset."));
      return;
    }

    Tileset->Tags.Add(DroneRsiCesiumTag);
    Tileset->SetGeoreference(TSoftObjectPtr<ACesiumGeoreference>(Georeference));
    Tileset->SetTilesetSource(ETilesetSource::FromCesiumIon);
    Tileset->SetIonAssetID(IonAssetId);
    Tileset->SetIonAccessToken(IonToken);
    Tileset->SetMaximumScreenSpaceError(8.0);
    Tileset->SetCreatePhysicsMeshes(true);
    Tileset->SetCreateNavCollision(true);
    Tileset->PreloadAncestors = true;
    Tileset->PreloadSiblings = true;
    Tileset->ForbidHoles = true;
    Tileset->FinishSpawning(FTransform::Identity);

    int32 HiddenBlocksGeometryCount = 0;
    for (TActorIterator<AStaticMeshActor> It(World); It; ++It) {
      AStaticMeshActor* MeshActor = *It;
      if (!IsValid(MeshActor)) {
        continue;
      }
      MeshActor->SetActorHiddenInGame(true);
      MeshActor->SetActorEnableCollision(false);
      ++HiddenBlocksGeometryCount;
    }

    UE_LOG(
        LogTemp,
        Display,
        TEXT("DroneRSI: spawned Cesium SF Unreal tileset asset %lld at lat %.6f lon %.6f with physics meshes enabled; hid %d Blocks static mesh actors."),
        IonAssetId,
        OriginLat,
        OriginLon,
        HiddenBlocksGeometryCount);
  }

  FDelegateHandle PostWorldInitHandle;
};

IMPLEMENT_PRIMARY_GAME_MODULE(FBlocksModule, Blocks, "Blocks");
