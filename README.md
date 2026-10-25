# PathLight GI — iluminação global + escuridão estilo Dark Souls 2 para o Dusklight

Mod nativo para o [Dusklight](https://github.com/TwilitRealm/dusklight) com duas partes que funcionam juntas:

1. **Iluminação global inspirada em path tracing**: raios Monte Carlo no espaço da tela, múltiplos
   rebotes, sangramento de cor, luz do céu, oclusão de contato, acumulação temporal e denoiser.
2. **Iluminação escura estilo Dark Souls 2**: masmorras e cavernas ficam realmente escuras e só são
   iluminadas por **tochas, velas, fogueiras, bombas, sua lanterna, a espada de luz e faíscas de
   espada**, com sombras suaves, tremulação de chama e luz que também "quica" nas paredes.

> **Não é path tracing de verdade** (como o RTX): o Dusklight roda sobre a geometria original, sem
> BVH. Os raios são traçados contra o buffer de profundidade da tela e não enxergam o que está fora
> do quadro ou atrás de objetos. As sombras das luzes têm a mesma limitação.

## Como funciona a parte escura

- O mod lê as **luzes reais do jogo** (`g_env_light.pointlight[]`): tochas, velas, luzes de sala,
  bombas, flechas de fogo etc. Nada é adivinhado pela imagem.
- A **lanterna** (e a espada de luz / olhos do lobo) usa a luz que o próprio Link emite.
- **Faíscas**: quando a espada acerta algo, um clarão curto de luz aparece no ponto de impacto.
- A escuridão depende do tipo de fase: `D_*` (templos, cavernas, grutas) ou `R_*` (interiores).
  A mudança entre claro e escuro leva ~1 s, sem salto.

## Opções (menu de Mods → PathLight GI → Open Controls)

| Opção | O que faz |
|---|---|
| Where | Onde escurecer: desligado / masmorras e cavernas (padrão) / + interiores / em todo lugar (mais escuro à noite) |
| Darkness | Quão escuro fica o ambiente sem luz (padrão 80%) |
| Light Strength / Radius | Brilho e alcance das luzes |
| Light Shadows | Objetos bloqueiam a luz e projetam sombras |
| Flame Flicker | Tremulação das chamas |
| Sword Sparks | Clarão ao acertar com a espada |
| Debug View | Mostra "Dynamic light only" e "Darkness scale" para ajustes |

## Instalação (jogador)

1. Baixe o `pathlight_gi.dusk` (veja abaixo como obter o seu).
2. Copie para a pasta de mods:
   - Windows: `%APPDATA%\TwilitRealm\Dusklight\mods`
   - Linux: `~/.local/share/TwilitRealm/Dusklight/mods`
   - macOS: `~/Library/Application Support/TwilitRealm/Dusklight/mods`
3. Ative **PathLight GI** no menu de Mods. Se atualizar de uma versão antiga, apague o `.dusk`
   antigo antes.

**Compatibilidade:** esta versão acessa dados do jogo e foi compilada contra o Dusklight **v2.0.3**.
Use o mesmo Dusklight (ou o mais próximo possível); em versões muito diferentes o mod pode falhar ao
carregar ou ler luzes erradas. Para outra versão, altere `DUSKLIGHT_VERSION` no `CMakeLists.txt`.

## Como obter o .dusk para o seu sistema

Um mod nativo precisa ser compilado para cada plataforma. Use o CI incluído:

1. Crie um repositório no GitHub e envie o conteúdo desta pasta (incluindo `.github` e
   `.gitattributes`).
2. A aba **Actions** compila para Linux, macOS, Windows, iOS e Android e junta tudo no artefato
   `mod-combined`. Ou baixe só o `mod-windows-amd64`.

## Compilar localmente

```
cmake -B build -G Ninja
cmake --build build
# resultado: build/mods/pathlight_gi.dusk
```

Requer CMake 3.26+, Ninja e um compilador C++20. O SDK e os cabeçalhos do Dusklight (v2.0.3) são
baixados automaticamente; use `-DDUSKLIGHT_DIR=/caminho/do/dusklight` para usar um checkout existente.

## Estrutura

- `src/mod.cpp` — integração com o renderer, opções e interface.
- `src/lights.cpp/.h` — leitura das luzes e do estado do jogo (tochas, lanterna, faíscas, tipo de fase).
- `res/*.wgsl` — shaders: `trace`, `temporal`, `denoise`, `composite` e `common` (luzes e escuridão).
- `tools/harness/` — banco de testes WebGPU em Chromium (cena Cornell box e caverna com tochas):
  `node tools/harness/run.js <pasta-saida> [opções-json]`.

## Estado dos testes

- Shaders validados no harness (Cornell box, caverna escura com duas tochas, sombras de luz).
- O código compila contra o Dusklight v2.0.3 e contra o branch principal.
- **Não foi testado dentro do jogo** (exige o disco original e uma GPU). Os campos do jogo lidos pelo
  mod foram conferidos no código-fonte do Dusklight, mas ajustes de brilho, raio e escuridão podem ser
  necessários. Se algo falhar, o log do Dusklight mostra mensagens com o prefixo
  `dev.community.pathlight_gi`.
