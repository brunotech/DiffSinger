from modules.commons.common_layers import *
from modules.commons.common_layers import Embedding
from modules.fastspeech.tts_modules import FastspeechDecoder, DurationPredictor, LengthRegulator, PitchPredictor, \
    EnergyPredictor, FastspeechEncoder
from utils.cwt import cwt2f0
from utils.hparams import hparams
from utils.pitch_utils import f0_to_coarse, denorm_f0, norm_f0
from modules.fastspeech.fs2 import FastSpeech2


from modules.fap.attribute_prediction_model import AGAP, f0_model_config, BGAP
from modules.fap.common import LinearNorm
from modules.fap.loss import AttributePredictionLoss


class FastspeechMIDIEncoder(FastspeechEncoder):
    def forward_embedding(self, txt_tokens, midi_embedding, midi_dur_embedding, slur_embedding):
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(txt_tokens)
        x = x + midi_embedding + midi_dur_embedding + slur_embedding
        if hparams['use_pos_embed']:
            if hparams.get('rel_pos') is not None and hparams['rel_pos']:
                x = self.embed_positions(x)
            else:
                positions = self.embed_positions(txt_tokens)
                x = x + positions
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, txt_tokens, midi_embedding, midi_dur_embedding, slur_embedding):
        """
        :param txt_tokens: [B, T]
        :return: {
            'encoder_out': [T x B x C]
        }
        """
        encoder_padding_mask = txt_tokens.eq(self.padding_idx).data
        x = self.forward_embedding(txt_tokens, midi_embedding, midi_dur_embedding, slur_embedding)  # [B, T, H]
        x = super(FastspeechEncoder, self).forward(x, encoder_padding_mask)
        return x


FS_ENCODERS = {
    'fft': lambda hp, embed_tokens, d: FastspeechMIDIEncoder(
        embed_tokens, hp['hidden_size'], hp['enc_layers'], hp['enc_ffn_kernel_size'],
        num_heads=hp['num_heads']),
}


class FastSpeech2FlowMIDI(FastSpeech2):
    def __init__(self, dictionary, out_dims=None):
        super().__init__(dictionary, out_dims)
        del self.encoder
        self.encoder = FS_ENCODERS[hparams['encoder_type']](hparams, self.encoder_embed_tokens, self.dictionary)
        self.midi_embed = Embedding(300, self.hidden_size, self.padding_idx)
        self.midi_dur_layer = Linear(1, self.hidden_size)
        self.is_slur_embed = Embedding(2, self.hidden_size)

        if hparams['use_pitch_embed']:
            self.pitch_flow_f0_uv_encoder = nn.Sequential(*[
                nn.Conv1d(256, 64, kernel_size=3, stride=1, padding=1, dilation=1),
                nn.ReLU(),
                nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1),
            ])
            f0_model_config['hparams']['bottleneck_hparams']['in_dim'] = self.hidden_size + 64
            self.pitch_flow = AGAP(**f0_model_config['hparams'])
            # self.pitch_flow = BGAP(**f0_model_config['hparams'])
            self.pitch_loss_fn = AttributePredictionLoss("f0", f0_model_config, 1.0)
            self.pitch_flow_uv_bias_encoder = nn.Sequential(*[
                LinearNorm(self.hidden_size + 64, 1), 
                # nn.ReLU()
                nn.Sigmoid()
            ])
            for p in self.pitch_flow_uv_bias_encoder.parameters():
                p.requires_grad = False

    def forward(self, txt_tokens, mel2ph=None, spk_embed=None,
                ref_mels=None, f0=None, uv=None, energy=None, skip_decoder=False,
                spk_embed_dur_id=None, spk_embed_f0_id=None, infer=False, enable_pitch_flow=False, **kwargs):
        ret = {}

        midi_embedding = self.midi_embed(kwargs['pitch_midi'])
        midi_dur_embedding, slur_embedding = 0, 0
        if kwargs.get('midi_dur') is not None:
            midi_dur_embedding = self.midi_dur_layer(kwargs['midi_dur'][:, :, None])  # [B, T, 1] -> [B, T, H]
        if kwargs.get('is_slur') is not None:
            slur_embedding = self.is_slur_embed(kwargs['is_slur'])
        encoder_out = self.encoder(txt_tokens, midi_embedding, midi_dur_embedding, slur_embedding)  # [B, T, C]
        src_nonpadding = (txt_tokens > 0).float()[:, :, None]

        # add ref style embed
        # Not implemented
        # variance encoder
        var_embed = 0

        # encoder_out_dur denotes encoder outputs for duration predictor
        # in speech adaptation, duration predictor use old speaker embedding
        if hparams['use_spk_embed']:
            spk_embed_dur = spk_embed_f0 = spk_embed = self.spk_embed_proj(spk_embed)[:, None, :]
        elif hparams['use_spk_id']:
            spk_embed_id = spk_embed
            if spk_embed_dur_id is None:
                spk_embed_dur_id = spk_embed_id
            if spk_embed_f0_id is None:
                spk_embed_f0_id = spk_embed_id
            spk_embed = self.spk_embed_proj(spk_embed_id)[:, None, :]
            spk_embed_dur = spk_embed_f0 = spk_embed
            if hparams['use_split_spk_id']:
                spk_embed_dur = self.spk_embed_dur(spk_embed_dur_id)[:, None, :]
                spk_embed_f0 = self.spk_embed_f0(spk_embed_f0_id)[:, None, :]
        else:
            spk_embed_dur = spk_embed_f0 = spk_embed = 0

        # add dur
        dur_inp = (encoder_out + var_embed + spk_embed_dur) * src_nonpadding

        mel2ph = self.add_dur(dur_inp, mel2ph, txt_tokens, ret)

        decoder_inp = F.pad(encoder_out, [0, 0, 1, 0])

        mel2ph_ = mel2ph[..., None].repeat([1, 1, encoder_out.shape[-1]])
        decoder_inp_origin = decoder_inp = torch.gather(decoder_inp, 1, mel2ph_)  # [B, T, H]

        tgt_nonpadding = (mel2ph > 0).float()[:, :, None]

        # add pitch and energy embed
        pitch_inp = (decoder_inp_origin + var_embed + spk_embed_f0) * tgt_nonpadding
        if hparams['use_pitch_embed']:
            pitch_inp_ph = (encoder_out + var_embed + spk_embed_f0) * src_nonpadding
            decoder_inp = decoder_inp + self.add_pitch(pitch_inp, f0, uv, mel2ph, ret, encoder_out=pitch_inp_ph, infer=infer, enable_pitch_flow=enable_pitch_flow, mel2f0_midi=kwargs['mel2f0_midi'])
        if hparams['use_energy_embed']:
            decoder_inp = decoder_inp + self.add_energy(pitch_inp, energy, ret)

        ret['decoder_inp'] = decoder_inp = (decoder_inp + spk_embed) * tgt_nonpadding

        if skip_decoder:
            return ret
        ret['mel_out'] = self.run_decoder(decoder_inp, tgt_nonpadding, ret, infer=infer, **kwargs)

        return ret
    
    def add_pitch(self, decoder_inp, f0, uv, mel2ph, ret, encoder_out=None, enable_pitch_flow=False, infer=False, **kwargs):

        decoder_inp = decoder_inp.detach() + hparams['predictor_grad'] * (decoder_inp - decoder_inp.detach())
        pitch_padding = mel2ph == 0
        ret['pitch_pred'] = pitch_pred = self.pitch_predictor(decoder_inp)
        if f0 is None or infer:
            f0 = pitch_pred[:, :, 0]
        if hparams['use_uv'] and (uv is None or infer):
            uv = pitch_pred[:, :, 1] > 0
        ret['f0_denorm'] = f0_denorm = denorm_f0(f0, uv, hparams, pitch_padding=pitch_padding)
        if pitch_padding is not None:
            f0[pitch_padding] = 0

        if enable_pitch_flow:
            diff_scale = 3.
            # diff_scale = 5.
            sigma_f0 = 0.3
            if infer: # inference stage
                z = torch.randn([decoder_inp.shape[0], 1, decoder_inp.shape[1]]).to(f0.device) * sigma_f0
                pred_pitch = f0_to_coarse(f0_denorm)  # start from 0
                pred_pitch_embed = self.pitch_embed(pred_pitch)
                pred_f0_encoding = self.pitch_flow_f0_uv_encoder(pred_pitch_embed.transpose(1,2))
                decoder_inp = decoder_inp.transpose(1,2)
                decoder_inp = torch.cat([decoder_inp, pred_f0_encoding], dim=1)
                spk_emb = torch.zeros([decoder_inp.shape[0], 0]).to(decoder_inp.device)
                pred_sample = self.pitch_flow.infer(z, decoder_inp, spk_emb).squeeze(1)

                if hparams.get("fit_midi_f0") is not None and hparams['fit_midi_f0'] is True:
                    mel2f0_midi = kwargs['mel2f0_midi']
                    f0_refined = denorm_f0(pred_sample * diff_scale, uv, hparams, pitch_padding=pitch_padding) 
                    f0_denorm_refined = mel2f0_midi + f0_refined
                else:
                    f0_denorm_refined = denorm_f0(pred_sample * diff_scale, uv, hparams, pitch_padding=pitch_padding)
                ret['f0_denorm'] = f0_denorm_refined
            else:
                gt_f0 = f0
                gt_uv = uv
                gt_f0 = denorm_f0(gt_f0, gt_uv, hparams, pitch_padding=pitch_padding)

                pred_f0 = pitch_pred[:, :, 0]
                pred_uv = pitch_pred[:, :, 1] > 0
                pred_f0 = denorm_f0(pred_f0, gt_uv, hparams, pitch_padding=pitch_padding)
                
                pred_pitch = f0_to_coarse(pred_f0)  # start from 0
                pred_pitch_embed = self.pitch_embed(pred_pitch)

                pred_f0_encoding = self.pitch_flow_f0_uv_encoder(pred_pitch_embed.transpose(1,2))
                decoder_inp = decoder_inp.transpose(1,2)
                decoder_inp = torch.cat([decoder_inp, pred_f0_encoding], dim=1)

                if hparams.get("fit_midi_f0") is not None and hparams['fit_midi_f0'] is True:
                    mel2f0_midi = kwargs['mel2f0_midi']
                    gt_sample = gt_f0 - mel2f0_midi
                    gt_sample[uv>0] = 0
                    # mask_2 = 
                    gt_sample = norm_f0(gt_sample, gt_uv, hparams, pitch_padding=pitch_padding) / diff_scale
                else:
                    gt_sample = norm_f0(gt_f0, gt_uv, hparams, pitch_padding=pitch_padding) / diff_scale

                # todo: gt_f0 - gt_midi, mask offset 50hz # librosa
                with torch.no_grad():
                    f0_uv_bias = self.pitch_flow_uv_bias_encoder(decoder_inp.transpose(1,2))
                    f0_uv_bias = - f0_uv_bias[..., 0]
                    zero_mask = torch.bitwise_or(uv<=0, pitch_padding)
                    f0_uv_bias[zero_mask] = 0
                    gt_sample = gt_sample + f0_uv_bias

                lens = (mel2ph != 0).long().sum(dim=-1)
                if lens[-1] < mel2ph.shape[-1]:
                    lens[-1] = mel2ph.shape[-1] # lens_max must match the mel dim 
                spk_emb = torch.zeros([decoder_inp.shape[0], 0]).to(decoder_inp.device)
                pitch_flow_ret = self.pitch_flow(decoder_inp, spk_emb, gt_sample, lens)
                loss_dict = self.pitch_loss_fn(pitch_flow_ret, lens, uv=None) # todo: 看看不mask会不会更好
                ret['loss_pitch_flow'] = loss_dict['loss_f0'][0]

        pitch = f0_to_coarse(f0_denorm)  # start from 0
        pitch_embed = self.pitch_embed(pitch)
        return pitch_embed